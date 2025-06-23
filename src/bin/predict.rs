use candle_core::{DType, Device, Result, Tensor};
use candle_nn::ops::sigmoid;
use candle_nn::var_builder::VarBuilder;
use candle_nn::{Linear, Module, VarMap};
use serde::Deserialize;
use std::fs::File;

const INPUT_DIM: usize = 6;
const HIDDEN_DIM: usize = 10;
const OUTPUT_DIM: usize = 1;

#[derive(Deserialize)]
struct NormParams {
    min: Vec<f32>,
    max: Vec<f32>,
}

fn normalize_input(input: Vec<f32>, min: &[f32], max: &[f32]) -> Vec<f32> {
    input
        .iter()
        .zip(min.iter().zip(max.iter()))
        .map(|(x, (&min, &max))| {
            if (max - min).abs() < 1e-6 {
                0.0 // 避免除以 0
            } else {
                (x - min) / (max - min)
            }
        })
        .collect()
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

    let file = File::open("norm.json")
        .map_err(|e| candle_core::Error::Msg(format!("Failed to open norm.json: {e}")))?;
    let norm: NormParams = serde_json::from_reader(file)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to parse norm.json: {e}")))?;

    let input_raw = vec![43.7606f32, 2f32, 4f32, 100f32, 2997341918f32, 104273706f32];

    let input_norm = normalize_input(input_raw, &norm.min, &norm.max);
    let input = Tensor::from_vec(input_norm, (1, INPUT_DIM), &device)?;

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&mut varmap, DType::F32, &device);
    let model = ResNetModel::new(&vb, INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)?;
    varmap.load("model.safetensors")?;
    println!("Loaded {} variables", varmap.all_vars().len());

    let logits = model.forward(&input)?;
    let prob = sigmoid(&logits)?;
    let predicted = prob.ge(0.5)?.to_dtype(DType::U32)?;

    println!(
        "Predicted class: {}",
        predicted.squeeze(0)?.squeeze(0)?.to_scalar::<u32>()?
    );
    println!(
        "Probability: {:.4}",
        prob.squeeze(0)?.squeeze(0)?.to_scalar::<f32>()?
    );

    Ok(())
}
