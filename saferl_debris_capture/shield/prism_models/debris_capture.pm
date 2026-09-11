// ============================================================
// debris_capture.pm
// PRISM probabilistic model for the debris capture task
// ============================================================
//
// Model type: Discrete-Time Markov Chain (DTMC)
// Abstracted state variables:
//   d  - discretised EE-to-debris distance bucket  (0 = contact, 4 = far)
//   f  - contact force level bucket                 (0 = safe, 2 = overload)
//   t  - discretised tumbling rate bucket           (0 = slow, 2 = fast)
//
// Safety specification (PCTL):
//   P<=0.05 [ F<=20 (f>=2) ]   -- collision prob within 20 steps <= 5%
//   P>=0.90 [ F<=50 (d=0) ]    -- capture prob within 50 steps  >= 90%
//
// ============================================================

dtmc

// ── Constants ─────────────────────────────────────────────────────────────
const int D_FAR       = 4;   // distance buckets: 0=contact, 4=far
const int F_SAFE      = 0;   // force buckets:    0=safe
const int F_HIGH      = 1;   //                   1=warning
const int F_OVERLOAD  = 2;   //                   2=collision
const int T_SLOW      = 0;   // tumble buckets:   0=slow (<5 deg/s)
const int T_MED       = 1;   //                   1=medium (5-15 deg/s)
const int T_FAST      = 2;   //                   2=fast   (>15 deg/s)

// Transition probabilities parameterised by tumbling rate
// (calibrated from offline rollouts of TumblingDebrisDynamicsNumpy)
const double P_COL_SLOW = 0.02;   // P(collision | slow tumble, close approach)
const double P_COL_MED  = 0.08;   // P(collision | medium tumble)
const double P_COL_FAST = 0.20;   // P(collision | fast tumble)
const double P_APP      = 0.70;   // P(approach succeeds | safe action)
const double P_STAY     = 0.25;   // P(distance unchanged)
const double P_RETREAT  = 0.05;   // P(unintentional retreat)

// ── Module: distance ──────────────────────────────────────────────────────
module distance
  d : [0..4] init 4;   // start far from debris

  // Safe approach action (action=0): move closer
  [approach] d>0 -> P_APP:(d'=d-1) + P_STAY:(d'=d) + P_RETREAT:(d'=min(d+1,4));

  // Hold position (action=1)
  [hold]     true -> 0.95:(d'=d) + 0.05:(d'=min(d+1,4));

  // Retreat (action=2): move away
  [retreat]  d<4  -> 0.90:(d'=d+1) + 0.10:(d'=d);

endmodule

// ── Module: force (collision risk) ────────────────────────────────────────
module force
  f : [0..2] init 0;

  // Force level depends on tumbling rate and approach distance
  [approach] (t=T_SLOW) & (d>1) -> (1-P_COL_SLOW):(f'=0) + P_COL_SLOW:(f'=1);
  [approach] (t=T_SLOW) & (d<=1) -> (1-2*P_COL_SLOW):(f'=0) + P_COL_SLOW:(f'=1) + P_COL_SLOW:(f'=2);

  [approach] (t=T_MED)  & (d>1) -> (1-P_COL_MED):(f'=0) + P_COL_MED:(f'=1);
  [approach] (t=T_MED)  & (d<=1) -> (1-P_COL_MED):(f'=1) + P_COL_MED:(f'=2);

  [approach] (t=T_FAST) & (d>1) -> (1-P_COL_FAST):(f'=0) + P_COL_FAST:(f'=1);
  [approach] (t=T_FAST) & (d<=1) -> 0.5:(f'=1) + 0.5:(f'=2);

  [hold]     true  -> 0.98:(f'=0) + 0.02:(f'=min(f+1,2));
  [retreat]  true  -> (f'=0);

endmodule

// ── Module: tumbling rate ─────────────────────────────────────────────────
module tumble
  t : [0..2] init 1;   // start at medium tumble

  // Tumble rate evolves stochastically (Markov chain over discretised states)
  [approach] (t=T_SLOW) -> 0.85:(t'=T_SLOW) + 0.15:(t'=T_MED);
  [approach] (t=T_MED)  -> 0.10:(t'=T_SLOW) + 0.80:(t'=T_MED) + 0.10:(t'=T_FAST);
  [approach] (t=T_FAST) -> 0.15:(t'=T_MED)  + 0.85:(t'=T_FAST);

  [hold]     (t=T_SLOW) -> 0.90:(t'=T_SLOW) + 0.10:(t'=T_MED);
  [hold]     (t=T_MED)  -> 0.10:(t'=T_SLOW) + 0.80:(t'=T_MED) + 0.10:(t'=T_FAST);
  [hold]     (t=T_FAST) -> 0.15:(t'=T_MED)  + 0.85:(t'=T_FAST);

  [retreat]  (t=T_SLOW) -> 0.90:(t'=T_SLOW) + 0.10:(t'=T_MED);
  [retreat]  (t=T_MED)  -> 0.15:(t'=T_SLOW) + 0.75:(t'=T_MED) + 0.10:(t'=T_FAST);
  [retreat]  (t=T_FAST) -> 0.20:(t'=T_MED)  + 0.80:(t'=T_FAST);

endmodule

// ── Labels ────────────────────────────────────────────────────────────────
label "collision"   = (f=F_OVERLOAD);
label "captured"    = (d=0) & (f<F_OVERLOAD);
label "warning"     = (f=F_HIGH);
label "far"         = (d=D_FAR);

// ── Properties (queries passed to PRISM at runtime) ──────────────────────
// P1: Probability of collision within 20 steps
property P_collision = P=? [ F<=20 "collision" ];

// P2: Probability of successful capture within 50 steps
property P_capture   = P=? [ F<=50 "captured"  ];

// P3: Probability of entering warning zone within 10 steps
property P_warning   = P=? [ F<=10 "warning"   ];
