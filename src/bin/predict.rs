use candle_core::{DType, Device, Result, Tensor};
use candle_nn::ops;
use candle_nn::var_builder::VarBuilder;
use candle_nn::{Linear, Module, VarMap};

const INPUT_DIM: usize = 4;
const HIDDEN_DIM: usize = 10;
const OUTPUT_DIM: usize = 1;

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

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&mut varmap, DType::F32, &device);
    let model = MLP::new(&vb)?;

    // Load model
    varmap.load("/home/eric-wcnlab/underdog/rust_ml/model.safetensors")?;
    println!("Loaded {} variables", varmap.all_vars().len());

    // Input data
    let input =
        Tensor::from_vec(vec![5.6, 3.0, 4.1, 1.3], (1, 4), &device)?.to_dtype(DType::F32)?;

    // Predicted (logit)
    let output = model.forward(&input)?;

    // Apply sigmoid to get probability
    let prob = ops::sigmoid(&output)?;
    let prob_val = prob.to_vec2::<f32>()?[0][0];
    let predicted_class = if prob_val >= 0.5 { 1 } else { 0 };

    println!("Predicted probability: {:.4}", prob_val);
    println!("Predicted class: {}", predicted_class);

    Ok(())
}
