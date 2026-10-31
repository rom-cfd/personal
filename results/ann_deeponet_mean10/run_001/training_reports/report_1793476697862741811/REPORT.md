# DeepONet training only

Branch: current Y9; trunk: current temperature and log pressure.
10 members, 10000 Adam updates/member, batch 512, fixed final checkpoint.
Learning rate: updates 1-6000 = .001; 6001-9000 = .0003; 9001-10000 = .0001.
Curves show actually observed mini-batch training loss, logged at update 1,
every 50 updates and the first update after each learning-rate change.
The final full-training-set MSE is measured separately over all 8815 pairs.
Neither quantity is validation/test error or evidence of rollout accuracy.
No Cantera comparison evaluation or test-based model selection was run.

Total member training wall seconds: 207.624
Final full-train MSE range: 1.796460e-04 to 2.423697e-04
