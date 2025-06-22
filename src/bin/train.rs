use candle_core::IndexOp;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::ops::sigmoid;
use candle_nn::optim::AdamW;
use candle_nn::var_builder::VarBuilder;
use candle_nn::{Linear, Module, Optimizer, VarMap, loss};
use csv::ReaderBuilder;

fn load_dataset(
    path: &str,
    feature_columns: &[usize],
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(std::fs::File::open(path)?);

    let mut features = Vec::new();
    let mut labels = Vec::new();

    for row in reader.records() {
        let record = row.map_err(|e| candle_core::Error::Msg(format!("CSV read error: {e}")))?;

        if record.get(0) == Some("task_id") {
            // Skip header row
            continue;
        }

        let num_cols = record.len();

        if num_cols < 2 {
            return Err(candle_core::Error::Msg(
                "CSV row has too few columns".to_string(),
            ));
        }

        for &col in feature_columns {
            let val = record
                .get(col)
                .ok_or_else(|| candle_core::Error::Msg(format!("Missing column {col} in row")))?
                .parse::<f32>()
                .map_err(|e| {
                    candle_core::Error::Msg(format!("Parse error in column {col}: {e}"))
                })?;
            features.push(val);
        }

        let label = record
            .get(num_cols - 1)
            .ok_or_else(|| candle_core::Error::Msg("Missing label column in row".to_string()))?
            .parse::<u32>()
            .map_err(|e| candle_core::Error::Msg(format!("Parse error in label: {e}")))?;
        labels.push(label);
    }

    let input_dim = feature_columns.len();

    let xs = Tensor::from_vec(features, (labels.len(), input_dim), device)?;
    let ys = Tensor::from_vec(labels.clone(), labels.len(), device)?.to_dtype(DType::U32)?;

    Ok((xs, ys))
}

struct ResNetBlock {
    linear1: Linear,
    linear2: Linear,
}

impl ResNetBlock {
    fn new(vb: &VarBuilder, dim: usize) -> Result<Self> {
        let linear1 = candle_nn::linear(dim, dim, vb.pp("linear1"))?;
        let linear2 = candle_nn::linear(dim, dim, vb.pp("linear2"))?;
        Ok(Self { linear1, linear2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.linear1.forward(x)?.relu()?;
        let h = self.linear2.forward(&h)?;
        x.add(&h)?.relu() // 殘差 + ReLU
    }
}

struct ResNetModel {
    input: Linear,
    block: ResNetBlock,
    output: Linear,
}

impl ResNetModel {
    fn new(
        vb: &VarBuilder,
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
    ) -> Result<Self> {
        let input = candle_nn::linear(input_dim, hidden_dim, vb.pp("input"))?;
        let block = ResNetBlock::new(&vb.pp("block"), hidden_dim)?;
        let output = candle_nn::linear(hidden_dim, output_dim, vb.pp("output"))?;
        Ok(Self {
            input,
            block,
            output,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.input.forward(x)?.relu()?;
        let h = self.block.forward(&h)?;
        self.output.forward(&h)
    }
}

fn main() -> Result<()> {
    let device = Device::Cpu;
    let path = "/home/eric-wcnlab/underdog/task_data.csv";
    let feature_columns = [7, 10, 11]; // Columns for features
    let (xs, ys) = load_dataset(path, &feature_columns, &device)?;

    print!(
        "Loaded {} samples with {} features\n",
        xs.dims()[0],
        xs.dims()[1]
    );

    println!("First xs row: {:?}", xs.i(0)?);

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&mut varmap, DType::F32, &device);
    let input_dim = feature_columns.len();
    let model = ResNetModel::new(&vb, input_dim, 10, 1)?;
    let mut opt = AdamW::new_lr(varmap.all_vars(), 1e-2)?;

    for epoch in 1..=100 {
        let logits = model.forward(&xs)?.clamp(-30.0f32, 30.0f32)?.squeeze(1)?;
        let ys_f32 = ys.to_dtype(DType::F32)?;
        let loss = loss::binary_cross_entropy_with_logit(&logits, &ys_f32)?;
        opt.backward_step(&loss)?;

        if epoch % 10 == 0 {
            let probs = sigmoid(&logits)?;
            let predicted = probs.ge(0.5)?;
            let correct = predicted
                .to_dtype(DType::U32)?
                .eq(&ys)?
                .to_dtype(DType::F32)?
                .sum_all()?
                .to_scalar::<f32>()?;
            let acc = correct / ys.dims()[0] as f32;
            println!(
                "Epoch {epoch:3}: loss = {:.4}, acc = {:.2}%",
                loss.to_scalar::<f32>()?,
                acc * 100.0
            );
        }
    }

    println!(
        "Saving {} variables to model.safetensors",
        varmap.all_vars().len()
    );
    varmap.save("model.safetensors")?;

    Ok(())
}
