use ndarray::{arr2, Array2};
use derivative::Derivative;

#[derive(Derivative)]
#[derivative(Debug, Clone)]
struct NeuralNetwork<'phi> {
    input: Array2<f64>,
    weights1: Array2<f64>,
    layer1: Array2<f64>,
    weights2: Array2<f64>,
    y: Array2<f64>,
    output: Array2<f64>,
    #[derivative(Debug="ignore")]
    phi: &'phi ActivationFunction,
}

pub struct ActivationFunction {
    apply: fn(&f64) -> f64,
    derivative: fn(&f64) -> f64,
}

pub static SIGMA: ActivationFunction = ActivationFunction {
    apply: |z| -> f64 {
        use std::f64::consts::E;
        1. / (1. + E.powf(-z))
    },
    derivative: |z| -> f64 {
        z * (1. - z)
    }
};

pub static RELU: ActivationFunction = ActivationFunction {
    apply: |z| z.max(0.),
    derivative: |z| {
        if z <= &0. { 0. }
        else { 1. }
    }
};

impl NeuralNetwork<'_> {
    pub fn new<'phi>(x: &Array2<f64>, y: &Array2<f64>, phi: &'phi ActivationFunction) -> NeuralNetwork<'phi> {
        use ndarray_rand::rand_distr::Uniform;
        use ndarray_rand::RandomExt;

        NeuralNetwork {
            input: x.clone(),
            weights1: Array2::random((x.dim().1, 4), Uniform::new(0., 1.)),
            layer1: Array2::zeros(y.dim()),
            weights2: Array2::random((4, 1), Uniform::new(0., 1.)),
            y: y.clone(),
            output: Array2::zeros(y.dim()),
            phi: &phi
        }
    }

    pub fn ff(self: &mut Self) {
        self.layer1 = self.input.dot(&self.weights1).map(self.phi.apply);
        self.output = self.layer1.dot(&self.weights2).map(self.phi.apply);
    }

    /*
        d_loss = 2 * (y-output) * d_sig(output)
        weights1 += self.input.T . ((d_loss . self.weights2.T) * d_sig(layer1))
        weights2 += = self.layer1.T . d_loss
    */
    pub fn bp(self: &mut Self) {
        let d_loss = 2. * (&self.y - &self.output) * self.output.map(self.phi.derivative);

        let d_weights2 = self.layer1.t().dot(&d_loss);
        let d_weights1 = self.input.t().dot(&(&
            (d_loss.dot(&self.weights2.t())) * &self.layer1.map(self.phi.derivative))
        );

        self.weights1 += &d_weights1;
        self.weights2 += &d_weights2;
    }
}

fn main() -> std::io::Result<()> {
    #[rustfmt::skip]
    let input = arr2(&[
        [0., 0., 1.],
        [0., 1., 1.],
        [1., 0., 1.],
        [1., 1., 1.]
    ]);
    let y = arr2(&[[0.], [1.], [1.], [0.]]);
    let mut nn = NeuralNetwork::new(&input, &y, &SIGMA);

    for _ in 0..1500 {
        nn.ff();
        nn.bp();
    }
    println!("{}", nn.output.t().row(0));

    Ok(())
}
