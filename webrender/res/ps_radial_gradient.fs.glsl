/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

float offset(int index) {
    return vOffsets[index / 4][index % 4];
}

float linearStep(float lo, float hi, float x) {
    float d = hi - lo;
    float v = x - lo;
    if (d != 0.0) {
        v /= d;
    }
    return clamp(v, 0.0, 1.0);
}

void main(void) {
    vec2 cd = vEndCenter - vStartCenter;
    vec2 pd = vPos - vStartCenter;
    float rd = vEndRadius - vStartRadius;

    // Solve for t in length(t * cd - pd) = vStartRadius + t * rd
    // using a quadratic equation in form of At^2 - 2Bt + C = 0
    float A = dot(cd, cd) - rd * rd;
    float B = dot(pd, cd) + vStartRadius * rd;
    float C = dot(pd, pd) - vStartRadius * vStartRadius;

    float x;
    if (A == 0.0) {
        // Since A is 0, just solve for -2Bt + C = 0
        if (B == 0.0) {
            discard;
        }
        float t = 0.5 * C / B;
        if (vStartRadius + rd * t >= 0.0) {
            x = t;
        } else {
            discard;
        }
    } else {
        float discr = B * B - A * C;
        if (discr < 0.0) {
            discard;
        }
        discr = sqrt(discr);
        float t0 = (B + discr) / A;
        float t1 = (B - discr) / A;
        if (vStartRadius + rd * t0 >= 0.0) {
            x = t0;
        } else if (vStartRadius + rd * t1 >= 0.0) {
            x = t1;
        } else {
            discard;
        }
    }

    oFragColor = mix(vColors[0],
                     vColors[1],
                     linearStep(offset(0), offset(1), x));

    for (int i=1 ; i < vStopCount-1 ; ++i) {
        oFragColor = mix(oFragColor,
                         vColors[i+1],
                         linearStep(offset(i), offset(i+1), x));
    }
}
