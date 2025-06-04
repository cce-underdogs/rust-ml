use candle_core::{DType, Device, Result, Tensor};
use candle_nn::optim::AdamW;
use candle_nn::var_builder::VarBuilder;
use candle_nn::{Linear, Module, Optimizer, VarMap, loss};
use csv::ReaderBuilder;

const INPUT_DIM: usize = 4;
const HIDDEN_DIM: usize = 16;
const OUTPUT_DIM: usize = 3;

fn load_iris_dataset(path: &str, device: &Device) -> Result<(Tensor, Tensor)> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(std::fs::File::open(path)?);
    let mut features = Vec::new();
    let mut labels = Vec::new();

    for row in reader.records() {
        let record = row.map_err(|e| candle_core::Error::Msg(format!("CSV read error: {e}")))?;
        for i in 0..INPUT_DIM {
            features.push(record[i].parse::<f32>().unwrap());
        }
        labels.push(record[INPUT_DIM].parse::<u32>().unwrap());
    }

    let xs = Tensor::from_vec(features, (labels.len(), INPUT_DIM), device)?;
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
    let (xs, ys) = load_iris_dataset("/home/eric-wcnlab/underdog/rust_ml/data/iris.csv", &device)?;

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&mut varmap, DType::F32, &device);
    let model = MLP::new(&vb)?;
    let mut opt = AdamW::new_lr(varmap.all_vars(), 1e-2)?;

    for epoch in 1..=100 {
        let logits = model.forward(&xs)?;
        let loss = loss::cross_entropy(&logits, &ys)?;
        opt.backward_step(&loss)?;

        if epoch % 10 == 0 {
            let predicted = logits.argmax(1)?;
            let correct = predicted.eq(&ys)?.sum_all()?.to_scalar::<u8>()?;
            let acc = correct as f32 / ys.dims()[0] as f32;
            println!(
                "Epoch {epoch:3}: loss = {:.4}, acc = {:.2}%",
                loss.to_scalar::<f32>()?,
                acc * 100.0
            );
        }
    }

    println!(
        "Saving {} variables to iris_model.safetensors",
        varmap.all_vars().len()
    );
    varmap.save("iris_model.safetensors")?;

    Ok(())
}
