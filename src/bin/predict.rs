use candle_core::{DType, Device, Result, Tensor};
use candle_nn::ops::sigmoid;
use candle_nn::var_builder::VarBuilder;
use candle_nn::{Linear, Module, VarMap};

const INPUT_DIM: usize = 3;
const HIDDEN_DIM: usize = 10;
const OUTPUT_DIM: usize = 1;

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
        x.add(&h)?.relu()
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

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&mut varmap, DType::F32, &device);
    let model = ResNetModel::new(&vb, INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)?;

    varmap.load("model.safetensors")?;
    println!("Loaded {} variables", varmap.all_vars().len());

    let input_raw = vec![100f32, 0f32, 3f32];
    let input = Tensor::from_vec(input_raw, (1, INPUT_DIM), &device)?;

    let logits = model.forward(&input)?;
    let prob = sigmoid(&logits)?;
    let predicted = prob.ge(0.5)?.to_dtype(DType::U32)?;

    println!(
        "Predicted class: {}",
        predicted.squeeze(0)?.to_scalar::<u32>()?
    );
    println!("Probability: {:.4}", prob.squeeze(0)?.to_scalar::<f32>()?);

    Ok(())
}
