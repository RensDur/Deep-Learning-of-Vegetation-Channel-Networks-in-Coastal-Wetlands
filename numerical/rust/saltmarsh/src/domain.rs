use rand::prelude::*;


pub struct Domain {
    // Domain properties
    width: usize,
    height: usize,
    dx: f64,
    dy: f64,

    // Domain fields
    h: Vec<Vec<f64>>,
    s: Vec<Vec<f64>>,
    u: Vec<Vec<f64>>,
    v: Vec<Vec<f64>>,
    b: Vec<Vec<f64>>,

    // Parameters
    h_in: f64,
    h_c: f64,
    h_0: f64,
    grav: f64,
    rho: f64,
    d_u: f64,
    nb: f64,
    nv: f64,
    k: f64,
    d_0: f64,
    p_d: f64,
    s_in: f64,
    q_s: f64,
    e_s: f64,
    p_e: f64,
    r: f64,
    q_q: f64,
    e_b: f64,
    d_b: f64,
    morphological_acc_factor: f64,
    p_est: f64

}


impl Domain {
    pub fn new(width: usize, height: usize, cell_size: f64) -> Domain {
        // Prepare data fields
        let h = vec![vec![0.0; width]; height];
        let s = vec![vec![0.0; width]; height];
        let u = vec![vec![0.0; width]; height];
        let v = vec![vec![0.0; width]; height];
        let b = vec![vec![0.0; width]; height];

        Domain {
            width,
            height,
            dx: cell_size,
            dy: cell_size,
            h,
            s,
            u,
            v,
            b,
            h_in: 1e-5,
            h_c: 1e-3,
            h_0: 0.02,    // Initial water thickness
            grav: 9.81,
            rho: 1000.0,   // Water density
            d_u: 0.5, // Turbulent Eddy velocity
            nb: 0.016,   // bed roughness for bare land
            nv: 0.2, // bed roughness for vegetated land
            k: 1500.0, // Vegetation carrying capacity
            d_0: 1e-7,    // Sediment diffusivity in absence of vegetation
            p_d: 0.99,    // fraction by which sediment diffusivity is reduced when vegetation is at carrying capacity
            s_in: 5e-9,   // Maximum sediment input rate
            q_s: 6e-4,    // water layer thickness at which sediment input is halved
            e_s: 2.5e-4,  // Sediment erosion rate
            p_e: 0.9, // Fraction by which sediment erosion is reduced when vegetation is at carrying capacity
            r: 3.2e-8,   // Intrinsic plant growth rate (=1 per year)
            q_q: 0.02,    // Water layer thickness at which vegetation growth is halved
            e_b: 1e-5,    // Vegetation erosion rate
            d_b: 6e-9,    // Vegetation diffusivity
            morphological_acc_factor: 44712.0, // Morphological acceleration factor, required for S and B
            p_est: 0.002 // Probability of vegetation seedling establishment
        }
    }

    //
    // Initialize domain with random vegetation tussocks
    //
    pub fn initialize(&mut self) {
        for y in 0..self.height {
            for x in 0..self.width {
                // Sedimentary elevation is zero everywhere
                self.s[y][x] = 0.0;

                // Flow velocities are zero everywhere
                self.u[y][x] = 0.0;
                self.v[y][x] = 0.0;

                // Uniform water thickness H0
                self.h[y][x] = self.h_0;

                // Vegetation density is zero everywhere, except for some randomly placed tussocks
                // Fill the array with uniform random numbers


            }
        }
    }




}