#[derive(Copy, Clone, Debug)]
pub struct Position {
    pub x: f64,
    pub y: f64,
    pub abs: f64,
}

impl Position {
    #[inline]
    fn set_abs(&mut self) {
        self.abs = (self.x * self.x + self.y * self.y).sqrt();
    }

    #[inline]
    pub fn predict(&mut self, vel: &Velocity, acc: &Acceleration, jerk: &Jerk, delta_t: f64) {
        let delta_t2 = delta_t * delta_t;
        let delta_t3 = delta_t2 * delta_t;
        self.x += vel.x * delta_t + 0.5 * acc.x * delta_t2 + 0.1666666 * jerk.x * delta_t3;
        self.y += vel.y * delta_t + 0.5 * acc.y * delta_t2 + 0.1666666 * jerk.y * delta_t3;
        self.set_abs();
    }

    #[inline]
    pub fn correct(&mut self, coe0: &Coefficient, coe1: &Coefficient, delta_t: f64) {
        let delta_t2 = delta_t * delta_t;
        self.x += 0.0416666 * coe0.x * delta_t2 + 0.008333333 * coe1.x * delta_t2;
        self.y += 0.0416666 * coe0.y * delta_t2 + 0.008333333 * coe1.y * delta_t2;
        self.set_abs();
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Velocity {
    pub x: f64,
    pub y: f64,
}

impl Velocity {
    #[inline]
    pub fn predict(&mut self, acc: &Acceleration, jerk: &Jerk, delta_t: f64) {
        let delta_t2 = delta_t * delta_t;
        self.x += acc.x * delta_t + 0.5 * jerk.x * delta_t2;
        self.y += acc.y * delta_t + 0.5 * jerk.y * delta_t2;
    }

    #[inline]
    pub fn correct(&mut self, coe0: &Coefficient, coe1: &Coefficient, delta_t: f64) {
        self.x += (0.166666 * coe0.x + 0.04166666 * coe1.x) * delta_t;
        self.y += (0.166666 * coe0.y + 0.04166666 * coe1.y) * delta_t;
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Acceleration {
    pub x: f64,
    pub y: f64,
}

impl Acceleration {
    #[inline]
    pub fn new(mass: f64, pos: &Position) -> Self {
        let inv_dist = 1.0 / pos.abs;
        let p = mass * inv_dist * inv_dist * inv_dist;
        Acceleration {
            x: -p * pos.x,
            y: -p * pos.y,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Jerk {
    pub x: f64,
    pub y: f64,
}

impl Jerk {
    #[inline]
    pub fn new(mass: f64, pos: &Position, vel: &Velocity) -> Self {
        let inv_dist = 1.0 / pos.abs;
        let p1 = mass * inv_dist * inv_dist * inv_dist;
        let p2 = 3.0 * (pos.x * vel.x + pos.y * vel.y) * inv_dist * inv_dist;
        Jerk {
            x: (p2 * pos.x - vel.x) * p1,
            y: (p2 * pos.y - vel.y) * p1,
        }
    }
}

#[derive(Clone, Copy)]
pub struct Coefficient {
    x: f64,
    y: f64,
}

impl Coefficient {
    #[inline]
    pub fn new(acc0: &Acceleration, acc1: &Acceleration, jerk0: &Jerk, jerk1: &Jerk, delta_t: f64) -> (Coefficient, Coefficient) {
        (Coefficient {
            x: - 6.0 * (acc0.x - acc1.x) - delta_t *(4.0 * jerk0.x + 2.0 * jerk1.x),
            y: - 6.0 * (acc0.y - acc1.y) - delta_t *(4.0 * jerk0.y + 2.0 * jerk1.y),
        },
        Coefficient {
            x: 12.0 * (acc0.x - acc1.x) + 6.0 * delta_t * (jerk0.x + jerk1.x),
            y: 12.0 * (acc0.y - acc1.y) + 6.0 * delta_t * (jerk0.y + jerk1.y),
        })
    }
}
