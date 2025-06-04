use candle_core::{DType, Device, Result, Tensor};
use candle_nn::ops;
use candle_nn::var_builder::VarBuilder;
use candle_nn::{Linear, Module, VarMap};

const INPUT_DIM: usize = 4;
const HIDDEN_DIM: usize = 16;
const OUTPUT_DIM: usize = 3;

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
    varmap.load("/home/eric-wcnlab/underdog/rust_ml/iris_model.safetensors")?;
    println!("Loaded {} variables", varmap.all_vars().len());

    // Input data
    let input =
        Tensor::from_vec(vec![6.4, 3.1, 5.5, 1.8], (1, 4), &device)?.to_dtype(DType::F32)?;

    // Predicted
    let output = model.forward(&input)?;
    let predicted = output.argmax(1)?;
    println!(
        "Predicted class: {}",
        predicted.squeeze(0)?.to_scalar::<u32>()?
    );

    let probs = ops::softmax(&output, 1)?;
    println!("Class probabilities: {:?}", probs.to_vec2::<f32>()?);

    Ok(())
}
