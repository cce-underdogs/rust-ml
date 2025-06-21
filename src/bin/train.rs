use candle_core::{DType, Device, Result, Tensor};
use candle_nn::optim::AdamW;
use candle_nn::var_builder::VarBuilder;
use candle_nn::{loss, Linear, Module, Optimizer, VarMap};
use csv::ReaderBuilder;

const INPUT_DIM: usize = 4;
const HIDDEN_DIM: usize = 16;
const OUTPUT_DIM: usize = 3;

fn load_iris_dataset(
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
        let num_cols = record.len();

        if num_cols < 2{
            return Err(candle_core::Error::Msg("CSV row has too few columns".to_string()));
        }

        for &col in feature_columns{
            let val = record.get(col)
                .ok_or_else(|| candle_core::Error::Msg(format!("Missing column {col} in row")))?;
                .map_err(|e|candle_core::Error::Msg(format!("Parse error in column {col}: {e}")))?;
            features.push(val);
        }

        let label = record.get(num_cols - 1)
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

struct MLP {
    l1: Linear,
    l2: Linear,
}

impl MLP {
    fn new(vb: &VarBuilder) -> Result<Self> {
        let l1 = candle_nn::linear(INPUT_DIM, HIDDEN_DIM, vb.pp("l1"))?;
        let l2 = candle_nn::linear(HIDDEN_DIM, OUTPUT_DIM, vb.pp("l2"))?;
        Ok(Self { l1, l2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.l1.forward(x)?.relu()?;
        self.l2.forward(&h)
    }
}

fn main() -> Result<()> {
    let device = Device::Cpu;
    let path = "/home/underdog/underdog/task_data.csv";
    let feature_columns = [2,7,8,9,10,11]; // Columns for features
    let (xs, ys) = load_iris_dataset(path,feature_columns, &device)?;

    println!("First X (features): {:?}", first_x.to_vec1::<f32>()?);
    println!("First Y (label): {:?}", first_y.to_scalar::<u32>()?);

    // let mut varmap = VarMap::new();
    // let vb = VarBuilder::from_varmap(&mut varmap, DType::F32, &device);
    // let model = MLP::new(&vb)?;
    // let mut opt = AdamW::new_lr(varmap.all_vars(), 1e-2)?;

    // for epoch in 1..=100 {
    //     let logits = model.forward(&xs)?;
    //     let loss = loss::cross_entropy(&logits, &ys)?;
    //     opt.backward_step(&loss)?;

    //     if epoch % 10 == 0 {
    //         let predicted = logits.argmax(1)?;
    //         let correct = predicted.eq(&ys)?.sum_all()?.to_scalar::<u8>()?;
    //         let acc = correct as f32 / ys.dims()[0] as f32;
    //         println!(
    //             "Epoch {epoch:3}: loss = {:.4}, acc = {:.2}%",
    //             loss.to_scalar::<f32>()?,
    //             acc * 100.0
    //         );
    //     }
    // }

    // println!(
    //     "Saving {} variables to iris_model.safetensors",
    //     varmap.all_vars().len()
    // );
    // varmap.save("iris_model.safetensors")?;

    Ok(())
}
