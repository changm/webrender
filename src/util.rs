/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use euclid::{Matrix4, Point2D, Rect, Size2D};
use num::Zero;
use time::precise_time_ns;

#[allow(dead_code)]
pub struct ProfileScope {
    name: &'static str,
    t0: u64,
}

impl ProfileScope {
    #[allow(dead_code)]
    pub fn new(name: &'static str) -> ProfileScope {
        ProfileScope {
            name: name,
            t0: precise_time_ns(),
        }
    }
}

impl Drop for ProfileScope {
    fn drop(&mut self) {
        /*
        if self.name.chars().next() != Some(' ') {
            let t1 = precise_time_ns();
            let ms = (t1 - self.t0) as f64 / 1000000f64;
            println!("{} {}", self.name, ms);
        }*/
    }
}

// TODO: Implement these in euclid!
pub trait MatrixHelpers {
    fn transform_rect(&self, rect: &Rect<f32>) -> Rect<f32>;
}

impl MatrixHelpers for Matrix4 {
    #[inline]
    fn transform_rect(&self, rect: &Rect<f32>) -> Rect<f32> {
        let top_left = self.transform_point(&rect.origin);
        let top_right = self.transform_point(&rect.top_right());
        let bottom_left = self.transform_point(&rect.bottom_left());
        let bottom_right = self.transform_point(&rect.bottom_right());
        let (mut min_x, mut min_y) = (top_left.x.clone(), top_left.y.clone());
        let (mut max_x, mut max_y) = (min_x.clone(), min_y.clone());
        for point in &[ top_right, bottom_left, bottom_right ] {
            if point.x < min_x {
                min_x = point.x.clone()
            }
            if point.x > max_x {
                max_x = point.x.clone()
            }
            if point.y < min_y {
                min_y = point.y.clone()
            }
            if point.y > max_y {
                max_y = point.y.clone()
            }
        }
        Rect::new(Point2D::new(min_x.clone(), min_y.clone()),
                  Size2D::new(max_x - min_x, max_y - min_y))
    }
}

pub trait RectHelpers {
    fn contains_rect(&self, other: &Rect<i32>) -> bool;
}

impl RectHelpers for Rect<i32> {
    fn contains_rect(&self, other: &Rect<i32>) -> bool {
        self.origin.x <= other.origin.x &&
        self.origin.y <= other.origin.y &&
        self.max_x() >= other.max_x() &&
        self.max_y() >= other.max_y()
    }
}

// Don't use `euclid`'s `is_empty` because that has effectively has an "and" in the conditional
// below instead of an "or".
pub fn rect_is_empty<N:PartialEq + Zero>(rect: &Rect<N>) -> bool {
    rect.size.width == Zero::zero() || rect.size.height == Zero::zero()
}

pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    (b - a) * t + a
}
