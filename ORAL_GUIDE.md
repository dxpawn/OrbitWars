# Orbit Wars — Oral Exam Guide

**Team:** Group 12, `IAI-RL-DirtyDozen` · **You:** Vũ Nguyên Đan (submitting representative)
**Two agents:** `ctx2` (a learned re-ranker) and `NeRL` (a model-predictive planner)
**Course:** Reinforcement Learning and Planning (UET–VNU)

This guide assumes you know nothing about RL and builds up from zero. Read it top to bottom once, then re-read Part 7 (likely questions) and Part 8 (cheat sheet) right before the exam.

**The single sentence to remember:**
> We built two agents. One *learns* which planet to attack from data (`ctx2`). The other doesn't learn at all — it *simulates the future* of the game and picks the move that helps us most (`NeRL`). NeRL scores higher (1141.8 vs ~1001), and the whole report is an honest comparison of **learning vs planning**.

---

## Part 1 — How Orbit Wars works (the game rules)

Orbit Wars is a real-time strategy game, similar in spirit to the old game *Galcon*. Think of it as a space version of "capture territory."

**The board:**
- A 100 × 100 square map.
- A **sun** of radius 10 sits in the centre. The sun is deadly: any fleet whose flight path crosses the sun is destroyed.
- **Planets** sit on the map. Some are **static** (fixed position); some **orbit** the sun (they move every turn).

**What planets do:**
- Each planet is owned by **you**, an **enemy**, or is **neutral** (unowned).
- Each planet has a **production** rate. Every turn, a planet you own makes new ships and adds them to its garrison.
- **Garrison** = the number of ships currently sitting on a planet (its defenders).

**What you do each turn:**
- You send **fleets** of ships from a planet you own to another planet to capture it.
- A fleet takes several turns to travel. **Fleet speed depends on fleet size — larger fleets move *faster*, up to a maximum of 6 units per turn** (a tiny 1-ship fleet crawls at 1 unit/turn). This is counter-intuitive, so remember it: a 1-ship fleet is slow, a big fleet is fast.
- If the target planet is **moving**, you must **lead the target** (aim where it *will be* when the fleet arrives, like a hunter leading a duck), and make sure the path doesn't cross the sun.

**How capture works:**
- A fleet captures a planet only if it **arrives with more ships than the defender has at the exact moment of arrival.**
- If it arrives with too few ships, the attack fails (the defenders shoot down the attackers; you've wasted those ships).
- Once you capture a planet, its production starts making ships for **you**.

**Winning:**
- A match lasts up to **500 turns**.
- You win by eliminating opponents or having the strongest position (most ships / planets / production) at the end. In our head-to-head tests NeRL often wipes out the opponent before turn 100.

**The observation (what the agent "sees" each turn):**
A snapshot of the world: every planet (owner, ship count, production, position), every fleet currently in flight, and incoming threats. The agent has a **1-second-per-turn** time limit to decide.

**Two game modes:**
- **2p** = 1 vs 1 (two players).
- **4p** = free-for-all (four players). 4p is harder to predict because there are more opponents, so both our agents use slightly more cautious settings in 4p.

**Why these rules matter for our project:** the rules are **fully known and deterministic** — if you know everyone's moves, you can compute exactly what happens next. That single fact is what makes NeRL (the planner) possible: you can *simulate* the future instead of *guessing* it.

---

## Part 2 — Key terminology (as used in the report)

Memorise these. Examiners often just point at a word in your report and ask "what's this?"

| Term | Plain meaning |
|---|---|
| **Agent** | A program that plays the game (decides moves). We have two: `ctx2` and `NeRL`. |
| **Heuristic** | A hand-written rule of thumb (e.g. "attack the closest weak planet"). No learning, no simulation — just fixed rules. |
| **Garrison** | The ships sitting on a planet (its defenders). |
| **Fleet** | A group of ships in flight from one planet to another. |
| **Production** | How many ships a planet makes per turn. |
| **Candidate** | One possible move: a (source planet → target planet) pair the agent is considering. |
| **Re-ranker** | A component that takes a list of candidate moves and **reorders them** by quality. `ctx2`'s neural network is a re-ranker — it does not invent moves, it just reorders the rule engine's list. |
| **Rule engine** | The hand-written part that handles all the physics and mechanics (aiming, fleet sizing, sun avoidance, combat). `ctx2` = rule engine + learned re-ranker. |
| **MLP (multi-layer perceptron)** | The simplest kind of neural network: layers of numbers multiplied and added, with a non-linear function (ReLU) in between. `ctx2`'s network is a small MLP. |
| **ReLU** | A simple non-linear function: `ReLU(x) = max(0, x)`. It lets a neural net learn non-straight-line relationships. |
| **Feature** | A single number describing the situation (e.g. "distance to target", "our ship share"). `ctx2` describes each candidate with 46 features. |
| **DeepSet / set pooling** | A trick so the network can judge a candidate **relative to the other candidates** in the same turn. We add the mean / max / min / std of all candidates' features as extra inputs. |
| **Permutation-invariant** | The network's answer doesn't depend on the *order* the candidates are listed in. DeepSet gives us this. |
| **Supervised learning** | Learning from labelled examples: "given these features, the right answer is *this score*." `ctx2`'s network is trained this way. |
| **Offline** | Training from a **fixed, pre-collected dataset** — no playing the game during training. (Opposite: **online** = learning while interacting with the game.) |
| **Model-predictive planner / MPC / receding horizon** | An agent that **simulates the next few turns** of the game and picks the move that gives the best predicted outcome, replanning every turn. NeRL is this. |
| **Horizon** | How many turns into the future the planner simulates (NeRL: 18 turns in 2p, 13 in 4p). |
| **Capture floor** | The **minimum** number of ships needed to capture a target (defenders + a small safety overhead + reinforcements arriving). Sending exactly this is efficient. |
| **Focus-fire** | Combining fleets from **several** planets that arrive on the **same turn**, so together they exceed the capture floor when no single planet could. |
| **Regroup** | Moving spare ships from safe planets toward planets that are about to be attacked. |
| **Snipe / anti-snipe** | A **snipe** is an enemy fleet timed to grab a planet exactly when it's weak (e.g. right after you capture it, before it builds defenders). **Anti-snipe** = detecting and defending against that. |
| **TrueSkill** | The rating system Kaggle's leaderboard uses. It's noisy — scores **drift by about ±30** even with no code change. This is why we don't over-read small score differences. |
| **ROI (return on investment)** | "Is this move worth it?" NeRL only fires a wave if its score beats a threshold (ROI = 1.5). |

---

## Part 3 — A crash course on the approaches (how people tackle a game like this)

There are four broad families. Know where our two agents sit.

**1. Pure heuristics (hand-written rules).**
"Always attack the nearest enemy planet I can beat." Fast, simple, no data, no learning. A strong baseline but rigid — it can't reason about the future. Our early `heuristic_v6` agent is this kind.

**2. Learning-based (machine learning).** Split into two important sub-types:
- **Supervised / imitation learning (offline):** collect examples of good moves, then train a model to copy them. No reward signal, no playing during training. **This is what `ctx2`'s network does.** Its ceiling is the quality of whatever produced the examples — it can copy the teacher but not surpass it.
- **Reinforcement learning (online):** the agent **plays the game**, gets **rewards** (win/lose, ships gained), and gradually improves its strategy through trial and error. No teacher needed; in principle it can discover strategies no human wrote down. (We have RL code in the repo — PPO + self-play league — from early exploration; see Part 6.)

**3. Planning / search with a known model.**
Because the game's rules are known, you can **simulate** the future and search for the best move — no learning at all. **This is what `NeRL` does** (model-predictive control). This is the "Planning" half of the course "Reinforcement Learning **and** Planning."

**4. Hybrids.**
Mix the above. **`ctx2` is a hybrid:** a hand-written rule engine does the mechanics, and a small learned model handles only the one hard judgement call ("which target is best"). The report's future-work idea is another hybrid: let learning *suggest* candidates and let planning *make the final pick*.

**Online vs offline (a common exam question):**
- **Offline** = learn from a fixed dataset collected in advance (ctx2's training).
- **Online** = learn/decide by interacting with the live environment (real RL training; also NeRL "plans online" every turn because it re-simulates from the current real state each turn).

**Where our two agents land:**

| | `ctx2` | `NeRL` |
|---|---|---|
| Family | Hybrid (rules + supervised learning) | Planning with a known model (MPC) |
| Learns? | Yes — a small network, trained **offline** | **No** — zero learned parameters |
| Core idea | Learn to *rank* targets | *Simulate* the future and pick the best launch |
| Leaderboard | ~1000.8 | **1141.8** (higher) |

---

## Part 4 — RL basics, refreshed (enough to defend the report)

You won't be quizzed as if this is an RL theory exam, but examiners may ask "what is an MDP?" or "what's the difference between SARSA and Q-learning?" Here are crisp, correct answers.

### 4.1 The MDP (Markov Decision Process) — the framework for all of RL
An MDP is the standard way to describe a decision problem. It has five parts:
- **States (S):** the situations the agent can be in (in Orbit Wars: the full board snapshot).
- **Actions (A):** what the agent can do (which fleets to send where).
- **Transitions (P):** the rules for how the state changes after an action. In Orbit Wars these are **known and exact** — that's the key.
- **Reward (R):** a number telling the agent how good an outcome is (e.g. +1 for winning).
- **Discount (γ, gamma):** how much we value future reward vs immediate reward (between 0 and 1).
- **"Markov" property:** the future depends only on the **current** state, not the full history. The current board tells you everything you need.

The agent's goal is to find a **policy** (π) — a rule mapping states → actions — that maximises total expected reward.

Two key quantities:
- **Value function V(s):** "how good is it to be in state s" (expected future reward from there).
- **Action-value Q(s, a):** "how good is it to take action a in state s, then play well after."

### 4.2 Model-free vs model-based (the big divide — and the heart of our report)
- **Model-based:** the agent **has or learns the rules** (transitions) and can simulate ahead. → **NeRL is model-based** (the model is *given* by the game).
- **Model-free:** the agent **does not** use a model of the rules; it just learns values or a policy from experience. → SARSA, Q-learning, DQN, policy gradient are all model-free.

### 4.3 Prediction vs control
- **Prediction:** estimate the value of a *fixed* policy ("how good is this way of playing?").
- **Control:** *find the best* policy ("what's the best way to play?").

### 4.4 Model-free prediction methods
- **Monte Carlo (MC):** play a **whole episode** to the end, then update value estimates using the actual total reward observed. Simple, unbiased, but you must wait until the game ends.
- **Temporal Difference (TD):** update **after every step** using a short-term estimate (bootstrapping — using your own current guess of the next state's value). Faster, works before the episode ends. TD(0) is the basic version.

### 4.5 Model-free control methods
- **SARSA (on-policy):** updates Q using the action the agent **actually took next**. "On-policy" = it learns about the policy it's currently following (including its exploration). Tends to be safer/more conservative.
- **Q-learning (off-policy):** updates Q using the **best possible** next action, regardless of what it actually did. "Off-policy" = it learns about the optimal policy while behaving more exploratively. The classic memory aid:
  - SARSA uses Q(next state, **next action actually taken**).
  - Q-learning uses **max** over Q(next state, **all actions**).
- **Exploration vs exploitation:** the agent must sometimes try non-greedy moves (explore) to discover better strategies, vs always taking the current best (exploit). Common method: **ε-greedy** (act randomly with small probability ε).

### 4.6 DQN (Deep Q-Network)
Q-learning where the Q-function is a **neural network** (needed when there are too many states to store in a table). Two famous tricks that make it stable:
- **Experience replay:** store past transitions in a buffer and train on random samples (breaks correlations).
- **Target network:** a slowly-updated copy of the network used to compute the learning target (prevents the network from chasing its own tail).
DQN is what beat Atari games from raw pixels.

### 4.7 Policy gradient methods
Instead of learning values and acting greedily, **directly learn the policy** (a network that outputs action probabilities) and nudge it toward actions that led to higher reward.
- **REINFORCE:** the basic version — increase the probability of actions from high-reward episodes.
- **Actor–Critic:** two parts — an **actor** (the policy) and a **critic** (a value estimate that reduces noise).
- **PPO (Proximal Policy Optimization):** a stable, popular actor-critic variant that limits how much the policy changes per update. **This is the RL algorithm in our repo** (`rl/ppo.py`).

### 4.8 MARL (Multi-Agent RL)
RL when there are **several learning agents** at once (Orbit Wars 2p/4p). The hard part: the environment is **non-stationary** — as opponents also change, the "rules of the game" from one agent's view keep shifting. A standard tool is **self-play / a league** (train against copies of yourself and past versions) — our repo has a `league` for this.

### 4.9 The honest punchline (very important for *this* exam)
Neither of our final agents is "textbook RL," and we say so openly:
- **`ctx2`'s learning is supervised (imitation) learning, not RL** — there's no reward and no exploration during training, just regression onto labels. It belongs to the "learning" side broadly.
- **`NeRL` is model-based planning (MPC)** — it belongs to the **"Planning"** half of the course. It uses no learning at all (hence the name *Not even RL*).
- So the project covers the **Planning** side strongly and the **learning** side via supervised imitation, and we **compare** the two. We also explored real RL (PPO + self-play league) early; it didn't beat the planner within the compute/time budget, so we didn't submit it.

---

## Part 5 — The two agents in detail (techniques + skills)

### 5.1 `ctx2` — hybrid rule engine + learned re-ranker

**One-line:** a hand-written engine does all the mechanics; a small neural network only decides **which target is best**.

**Why split it this way?** Principle: *don't use machine learning where a reliable rule already works.* Spend the model only on the single hardest judgement — target selection — and let rules handle everything that's mechanical.

**The per-turn pipeline (6 steps):**
1. **Parse** the observation into planets, fleets, threats.
2. **Generate candidates:** for each owned planet, the engine lists reachable targets, pre-sorted by a distance heuristic. Each candidate gets an index `idx` (its heuristic rank). An `expand=10` factor keeps extra candidates before trimming.
3. **Build features:** turn each (source, target) into a **46-number feature vector**.
4. **Score with the network:** add DeepSet set-statistics (mean/max/min/std of all candidates) → **230 inputs**, run the MLP, get a quality score per candidate.
5. **Re-rank:** combine the heuristic rank and the learned score:
   `priority = idx − bonus · sigmoid(score)` (bonus = 1.45 in 2p, 1.25 in 4p).
   A high learned score lifts a candidate up the list; `bonus` caps how far the network can override the rules.
6. **Execute:** the engine computes how many ships to send (enough to win *and* hold), leads the moving target, avoids the sun, and runs a defensive reinforcement pass.

**The network (architecture):**
- A 3-layer MLP: **230 → 128 → 128 → 1**, with ReLU between layers.
- Input = 46 base features + 4×46 set-statistics (mean, max, min, std) = **230**.
- Output = one quality score per candidate.
- Exported to **plain Python** (no PyTorch at game time), so it runs in ~0.1 s/turn.
- Two separate weight files for 2p and 4p (the game looks different with 2 vs 4 players).

**The 46 features (you don't need all of them — know the categories):**
- **Global state (0–11):** game progress, who's leading, our ship/planet/production share, momentum, whether we're being targeted.
- **Source & target (12–19):** source ships/production/safety, target production, is the target moving, whose planet it is.
- **Capture & threat (20–31):** predicted owner and defenders at arrival, threat arriving after we capture, ETA, ships needed, can we win.
- **Strategy & terrain (32–45):** garrison left after capture, return-on-investment, distance to enemy front, how surrounded we are, convergence threats.
- A small **history buffer** (5–10 turns) gives "trend" features (momentum, aggression trend) so the net senses the game *phase*, not just a snapshot.

**`ctx2`'s skills / strategies (from the rule engine):**
- **Lead-aiming** moving targets (aim where the planet will be).
- **Sun avoidance** (never route a fleet through the sun).
- **Right-sized capture** (send enough to win *and* survive the counter).
- **Snipe / counter-snipe detection** (anti-snipe): spots enemy fleets timed to grab a planet and reacts.
- **Reinforcing weak planets** and **late recapture** in 2p.
- **Short-horizon look-ahead in 4p** (a 7-turn projection to adjust ranking).

**How it scores:** ~**1000.8** on the public leaderboard (was ~1067 earlier — TrueSkill drifts).

### 5.2 `NeRL` — model-predictive planner ("Not even RL")

**One-line:** it has **no neural network and no learned weights**. It builds an exact simulation of the game, predicts the next ~18 turns, and picks the launches that improve the predicted outcome the most. (It uses `torch` only as a fast calculator for the arrays — *not* as a learned model.)

**Why it's possible:** the game's rules are known and deterministic, so the future can be **computed**, not guessed.

**The per-turn pipeline:**
1. **Parse** the observation into tensors (planets, fleets).
2. **Build the movement model:** predict every planet's position over the horizon, and track all fleets in flight.
3. **Distance cache:** precompute "where will planet B be when a fleet from planet A could reach it" — the geometrically correct distance for checking if a fleet can intercept a *moving* planet in time.
4. **Garrison prediction:** for each future turn, compute each planet's predicted (owner, ships) using an **exact** step-by-step recurrence — production, fleet arrivals, and combat all resolved in order.
5. **Plan waves** (the core, below).
6. **Emit** the chosen launches.

**The core — wave planning and scoring:**
- **Shortlists:** pick promising **sources** (our planets with enough ships) and **targets** — both *offensive* (enemy/neutral planets) and *defensive* (**flip targets** = our own planets predicted to be lost soon, ranked by urgency).
- **Fleet sizing (`safe_drain`):** send the most ships a source can spare while still holding itself over the horizon.
- **Capture floor:** the **minimum** ships to take a target = ⌈defenders at arrival + overhead + reinforcements⌉. A move is kept only if it clears this floor. → efficient ship use, never overspend.
- **Focus-fire:** if no single planet can clear the floor, **combine** fleets from several planets that **arrive on the same turn**. This coordinated strike is the strongest feature.
- **Marginal competitive score** — how each move is judged:
  `s(c) = Δη_me − Σ(Δη_others)`
  In words: a move's value = (how many net ships it gains **for us**) **minus** (how many net ships it gains **for all opponents**). "Net ships" = produced minus lost over the horizon. This is a **one-step look-ahead over the exact simulator** — it rewards moves that help us and hurt opponents.
- **Greedy selection:** repeatedly take the highest-scoring non-conflicting wave, up to `max_waves_per_turn` (6 in 2p), only if its score beats the **ROI threshold** (1.5).
- **Regroup:** finally, shuffle spare ships from safe planets toward threatened ones, following an "enemy-pressure gradient."

**`NeRL`'s skills / strategies:**
- **Efficient capture** (capture floor — never wastes ships).
- **Coordinated multi-planet strikes** (focus-fire).
- **Defensive repositioning** (regroup toward threatened planets).
- **Anti-snipe / defence:** predicts which of *our* planets will be lost and prioritises saving them (the defensive "flip target" shortlist).
- **Exact orbital aiming** (intercept-angle solver + sun-occlusion check).

**Settings (configs):**
- **2p:** horizon 18 turns, up to 6 waves/turn, ROI 1.5, focus-fire on (up to 4 strike sources).
- **4p:** shorter horizon (13), fewer sources/targets — because 4-player games branch more and the future is less predictable.

**How it scores:** **1141.8** on the public leaderboard — above `ctx2`. In direct duels it **crushes** our other agents (beat heuristic_v6 1211–0, beat ctx2 1478–0, both before turn 100).

### 5.3 Why NeRL beats ctx2 (the report's main argument)
Three reasons — be ready to say these:
1. **It optimises the real outcome, not a label.** ctx2 learns to *match a teacher's preference*; a supervised model can't beat its teacher (the **imitation ceiling**). NeRL optimises the *simulated game result* directly, so it has no teacher to be limited by.
2. **It uses ships efficiently** (capture floor sends the smallest winning fleet, freeing ships for more targets).
3. **It coordinates** (focus-fire across planets — something a re-ranker that scores each move alone cannot represent).

Plus: ctx2 suffers **distribution shift** — it was trained on one set of situations but at game time visits its *own* situations, and small errors **compound**. NeRL re-plans from the true state every turn, so it has no train/test gap. (The textbook fix for ctx2's problem is **DAgger** — relabel the states the agent actually visits and retrain.)

---

## Part 6 — How the two agents were developed (reproducibility)

This is the section that answers "show me your work" — including "where's your notebook?"

**Workflow choice:** we used **reproducible Python scripts and version-controlled modules**, not Jupyter notebooks. (See the no-notebook justification below — it's a deliberate, defensible choice.)

**`NeRL` (the planner) — no training at all.**
- It's a single self-contained file (`agents/NeRL.py`).
- There is **no dataset, no training, no learned weights** — nothing to "train." The agent *is* the algorithm.
- It is **deterministic for a given seed**, so re-running a match reproduces the result exactly.

**`ctx2` (the learned re-ranker) — a 4-step, one-command pipeline.**
1. **Collect:** run the agent over ~250 games (2p and 4p) against a mixed pool of opponents; record each decision's 46 features and the teacher's quality score. → ~510,000 labelled candidates (~85 MB `.npz`).
2. **Train:** `python -m rl.distill_train_ctx --data ... --hidden 128 --epochs 60`. One command. It:
   - joins the 46 features with each group's mean/max/min/std → 230 inputs,
   - splits 90/10 **by decision group** (not by row) so train and validation don't leak,
   - fits with MSE loss and Adam, **seeded** (`torch.manual_seed(0)`), ~1 minute on GPU,
   - prints the metrics (`val_R2`, held-out top-1 agreement).
3. **Validate:** held-out **top-1 agreement** — in each held-out decision, does the model's top pick match the teacher's? Result: **81.3% (2p), 84.8% (4p)**, R² 0.917 / 0.933.
4. **Export:** write the weights to plain Python (`student_weights_{2p,4p}.py`) so the submission needs no PyTorch.

**Why no notebook (your prepared answer):**
> "We deliberately used reproducible scripts, not a notebook. One agent (NeRL) is a pure planner with **nothing to train** — a notebook would have no content. The other agent's training is a **single seeded command** that reproduces our exact reported numbers, which is *more* reproducible than a notebook (notebooks suffer from hidden state and out-of-order cells). Everything lives in version-controlled modules plus an evaluation harness, and the project requirements don't ask for a notebook."

**Where it all lives:** GitHub — `https://github.com/dxpawn/OrbitWars`. Both agents, the training pipeline, the evaluation harness, and the report source.

---

## Part 7 — Likely oral questions and short answers

Practice saying these out loud.

**Q: Is this Reinforcement Learning?**
> Partly, and we're upfront about it. NeRL is **model-based planning** (the Planning half of the course) — no learning. ctx2's model is **supervised/imitation learning**, not RL. We also tried real RL (PPO + self-play) early; it didn't beat the planner in our budget, so we didn't submit it. The report's contribution is the **comparison of learning vs planning**.

**Q: Why is the agent called "NeRL"?**
> *Not even RL* — an honest name. It contains no neural network and no learned weights. It plays well purely by simulating the game's known rules and choosing the best launch.

**Q: What's the difference between your two agents?**
> ctx2 *learns* to rank targets from data; NeRL *simulates* the future and computes the best move. NeRL scores higher (1141.8 vs ~1001) and wins head-to-head, because it optimises the real outcome instead of imitating a teacher.

**Q: Why does NeRL beat ctx2?**
> Three reasons: it optimises outcomes not labels (no imitation ceiling), it uses ships efficiently (capture floor), and it coordinates strikes (focus-fire). Also it re-plans from the true state each turn, so no distribution shift.

**Q: What is the imitation ceiling / distribution shift?**
> Imitation ceiling: a supervised model can only copy its teacher, never beat it. Distribution shift: it's trained on one set of states but at game time it visits its own states, and small errors compound. The standard fix is DAgger.

**Q: What's a DeepSet and why use it?**
> A way to let the network judge a candidate **relative to the other candidates** that turn, by adding their mean/max/min/std as extra inputs. It's permutation-invariant (order doesn't matter) and handles a varying number of candidates.

**Q: Explain SARSA vs Q-learning. (general RL)**
> Both learn Q-values. SARSA is on-policy — it updates using the action actually taken next. Q-learning is off-policy — it updates using the *best* next action (a max). Q-learning learns the optimal policy while behaving exploratively.

**Q: Where's your notebook? / Why no notebook?**
> (Use the prepared answer in Part 6.)

**Q: Your score is only mid-table (12/19 in class). Why?**
> We're honest about it: NeRL beats our *own* earlier agents decisively, but across all 4106 teams it's mid-field, and several class teams use similarly strong public agents. The value here is the **analysis of *why* planning beats learning**, not the raw rank. Also TrueSkill drifts ±30, so small gaps aren't meaningful.

**Q: What is the capture floor?**
> The minimum ships needed to take a planet = defenders at arrival + a small overhead + reinforcements arriving, rounded up. Sending exactly that avoids wasting ships.

**Q: What is focus-fire?**
> Combining fleets from several planets that arrive on the *same* turn, so together they exceed the capture floor when no single planet could.

**Q: How does an agent hit a moving planet?**
> Lead-aiming: predict where the planet will be when the fleet arrives (a fixed-point/intercept-angle calculation), and check the path doesn't cross the sun.

**Q: What was your contribution specifically?** (you, Vũ Nguyên Đan)
> Team lead; built the ctx2 agent (the hybrid rule-engine + learned re-ranker) and wrote the report. (Phạm Tiến Dũng built NeRL; Lê Hồng Anh contributed in week 1.)

**Q: If you had more time, what next?**
> Opponent modelling in NeRL (a shallow adversarial roll-out), replacing greedy wave selection with a small beam search, and a true hybrid: use ctx2's learned score to break ties inside NeRL's shortlist — learning suggests, planning decides.

---

## Part 8 — Cheat sheet (numbers + 60-second pitch)

**Numbers to have cold:**
- ctx2 score: **~1000.8** (earlier ~1067). NeRL score: **1141.8**.
- Class rank: **12 of 19** `IAI-RL-*` teams. Global: **636 of 4106**.
- ctx2 network: **230 → 128 → 128 → 1** MLP; **46** base features; trained on **~510,000** candidates; top-1 **81.3% / 84.8%** (2p/4p); R² **0.917 / 0.933**; inference ~**0.1 s/turn**.
- NeRL: horizon **18** (2p) / **13** (4p); up to **6** waves/turn; ROI threshold **1.5**; score `s(c) = Δη_me − Σ Δη_others`.
- Game: **100×100** board, sun radius **10**, max fleet speed **6** (bigger fleet = faster, 1-ship = speed 1), **500** turns, **1 s/turn** limit.
- Head-to-head: NeRL beat heuristic **1211–0**, beat ctx2 **1478–0** (both before turn 100).

**Your 60-second pitch:**
> "Orbit Wars is a real-time strategy game: planets orbit a sun, produce ships, and you send fleets to capture them. We built two agents and compared **learning vs planning**.
> The first, **ctx2**, is a hybrid: a hand-written engine handles all the mechanics — aiming, fleet sizing, sun avoidance — and a small neural network handles only the hard call, *which target to attack*. It reads 46 features per candidate plus set-statistics, runs a 230-128-128-1 MLP, and is trained offline on about half a million examples. It scores around 1001.
> The second, **NeRL** — *Not even RL* — has no learning at all. It simulates the next ~18 turns exactly and picks the launches that most improve our predicted ship advantage, using a capture floor for efficiency and focus-fire to coordinate strikes. It scores 1141.8 and beats ctx2 head-to-head.
> Our conclusion is the classic trade-off: when you have an exact model of the game, **planning with it beats learning to imitate** — because the planner optimises the real outcome instead of copying a teacher. The best next step is to combine them: let learning suggest moves and let planning make the final choice."

---

---

## Part 9 — NeRL deep dive (standalone)

*Read this as your complete NeRL reference. It repeats nothing from Part 5.2 that you need to re-read — it goes deeper and folds in everything we worked through, including the parts that confused people. This is the depth you need to cover NeRL if your teammate freezes.*

### 9.1 The essence (one paragraph)
NeRL is a **model-predictive planner**. It has **no neural network and no learned weights** (it uses `torch` only as a fast calculator for arrays). Every turn it builds an **exact simulation** of the game, predicts the next ~18 turns, and chooses the launches that most improve our predicted ship advantage — then it executes only this turn's launches and **re-plans from scratch next turn**. The name *"Not even RL"* is the honest label: it belongs to the **Planning** half of the course, not the learning half.

### 9.2 NeRL-specific terms (so this section stands alone)

| Term | Plain meaning |
|---|---|
| **Model-predictive control (MPC) / receding horizon** | Plan a few turns ahead, do only the first move, then re-plan next turn. The look-ahead window slides forward with you. **It is forward simulation, not backward** (see 9.3). |
| **Horizon (H)** | How many turns ahead it simulates: **18 in 2p, 13 in 4p**. |
| **Garrison prediction** | For each future turn, compute each planet's predicted (owner, ship count) by an exact step-by-step recurrence: production + fleet arrivals + combat. |
| **Competitive score `s(c)`** | How a candidate launch is judged: `s(c) = Δη_me − Σ Δη_others`. Net ships it gains us, minus net ships it gains every opponent. |
| **Capture floor** | The **minimum** ships needed to take a target: `⌈defenders at arrival + overhead + reinforcements⌉`. Never overspend. |
| **Focus-fire** | Combine fleets from several planets that **arrive on the same turn**, so together they clear the floor when no single planet can. |
| **`safe_drain`** | The most ships a source can send while still holding **itself** over the horizon. |
| **Regroup** | Move spare ships from safe planets toward threatened friendly planets, following an "enemy-pressure gradient." |
| **ROI threshold** | A **minimum-score cutoff** (1.5), **not a ratio** (see 9.4). A wave fires only if its `s(c)` beats it. |
| **Reachability / intercept angle** | Geometry: can a fleet actually reach a *moving* planet in time, around the sun? The intercept-angle solver gives the aim point. |

### 9.3 "Receding horizon" — the thing people get wrong
It is **forward** simulation. "Receding" describes how the *window moves*, not the direction. Each turn:
1. From the **real current state**, simulate H turns forward.
2. Pick this turn's best launches.
3. **Execute only this turn — throw the rest of the plan away.**
4. Next turn, the game has advanced and opponents have really moved; **re-simulate H turns forward from the new state.**

So the look-ahead always stays H turns ahead of "now," sliding forward like a horizon you walk toward but never reach. The benefit over planning all 500 turns once: every turn you fold in **new information** instead of following a stale plan. "Receding horizon" = "MPC" = same thing.

### 9.4 Wave planning and scoring (the core)
1. **Shortlists.** Pick promising **sources** (our planets with ≥ `min_ships_to_launch` = 4 ships) and **targets** — both *offensive* (enemy/neutral planets) and *defensive* **flip targets** (our own planets predicted to be lost soon, ranked by urgency).
2. **Fleet sizing = `safe_drain`** (send max while still holding the source).
3. **Capture floor gate.** Keep a candidate only if its size clears `⌈defenders + overhead + reinforcements⌉`.
4. **Focus-fire.** If no single source clears the floor, combine sources that arrive the same turn.
5. **Score = `s(c) = Δη_me − Σ Δη_others`** — a **one-step look-ahead over the exact simulator**: reward launches that raise our net ships and lower opponents'.
6. **Greedy selection + ROI gate.** Repeatedly take the highest-scoring non-conflicting wave, up to **6** per turn, **only if `s(c) > 1.5`**.
7. **Regroup.** Finally, shuffle spare ships toward threatened planets.

⚠️ **The ROI clarification (a confusion point):** the "ROI threshold" is **not** a return-on-investment ratio. In the code (`_greedy_select`) it's literally `fired = best_score > roi_threshold` — a **minimum-score cutoff**. So "ROI = 1.5" means *"only fire a wave if it nets more than +1.5 competitive ships."* (Separately, **ctx2** has a real ratio feature called `target_roi = production ÷ ships_needed` — don't confuse the two.)

### 9.5 Configs (2p vs 4p)

| Setting | 2p | 4p | Why 4p is more cautious |
|---|---|---|---|
| Horizon | 18 | 13 | 4 players branch more; long look-ahead is less reliable |
| Waves/turn | 6 | 6 | |
| ROI threshold | 1.5 | 1.5 | |
| Focus-fire sources | up to 4 | up to 3 | fewer coordinated strikes |
| Source/target shortlists | wider (12 / 4) | narrower (6 / 2) | less compute per branch |

### 9.6 Why NeRL beats ctx2 (this is YOUR section's crux)
Three structural advantages plus one:
1. **Optimises outcomes, not labels.** `s(c)` is based on the real game dynamics; ctx2 only imitates a fixed teacher and is capped by the **imitation ceiling**.
2. **Efficient ship use.** The capture floor sends the smallest winning fleet, freeing ships for more targets — an efficiency the feature-based re-ranker only approximates.
3. **Coordination.** Focus-fire runs multi-planet strikes a per-candidate re-ranker can't represent.
4. **No distribution shift.** NeRL re-plans from the true state every turn, so there's no train/test gap and no compounding error. ctx2 visits its own states at play time and small errors snowball.

**Evidence:** head-to-head it beat `heuristic_v6` **1211–0** and ctx2 **1478–0**, both before turn 100. On the leaderboard it scores **1141.8** vs ctx2's ~1001.

### 9.7 Limitations and future work
**The big limitation: NeRL does not model the opponent.** It assumes rivals **launch nothing new** beyond fleets already in flight — it plans as if the enemy is frozen. So a move can look great in simulation because the sim assumes no counter-attack, then fail in the real game. (ctx2 has the same blind spot.)

Three proposed fixes (future work — not built):
1. **Shallow adversarial roll-out** — for each candidate, also simulate the strongest rival's *best reply*, and only pick moves that survive it. ("Shallow" = one layer deep, not a full game tree.) **This is your strongest "what would you improve" answer.**
2. **Beam search instead of greedy** — instead of always grabbing the single best wave, keep the top few wave-combinations and pick the best overall set. Better cross-wave coordination.
3. **ctx2 as a tie-breaker inside NeRL** — when the planner rates two moves equally, let ctx2's learned score break the tie. *Learning suggests, planning decides.* This finally uses ctx2's learned priors in the one spot where its compounding-error weakness can't hurt (the planner re-decides every turn anyway).

### 9.8 Strategic context (the 3am findings — important)
- **The popular public "1300" agent (`IMBETTER_LB1300`) is the SAME planner family as NeRL.** Its core lists the identical techniques: competitive net-ship-delta scorer, capture-floor sizing, `safe_drain`, greedy selection, pressure-gradient regroup. So **"I have a planner" does NOT make you stand out** — much of the class will submit the same kind of planner, tuned better.
- **Your differentiation is the *comparison*, not the agent.** Two paradigms (learned ctx2 + planner NeRL) plus an honest analysis of why planning wins. The copiers have one agent and no analysis.
- **The leaderboard proves your thesis — use it:** the strongest public agents (1221, 1300) are **planners**; the imitation-based ones rank **lower**. Your two agents reproduce that ordering in miniature. This is a true, sourcing-safe, A-grade point.
- **Honesty line:** the planner's techniques (MPC, capture floor, competitive scoring) are **standard planning ideas, not novel inventions** — don't claim otherwise. Claim the **comparison and the engineering**. Be modest about the agent, confident about the thinking.

### 9.9 NeRL Q&A (drill these)

**Q: Is NeRL reinforcement learning?**
> No. It's model-based **planning** (MPC) — the Planning half of the course. No learning, no weights. It plans well purely by simulating the game's known rules.

**Q: Then why is it in an RL course?**
> The course is "Reinforcement Learning **and** Planning." NeRL is the planning side; ctx2 is the learning side. The report compares them.

**Q: How does it decide what to attack?**
> It simulates the next 18 turns and scores each candidate launch by `s(c) = Δη_me − Σ Δη_others`, then greedily picks the best non-conflicting waves above an ROI cutoff.

**Q: What does "receding horizon" mean — is it backward?**
> Forward. It plans H turns ahead, executes only this turn, then re-plans from the new state next turn. The look-ahead window slides forward. (See 9.3.)

**Q: How is the ROI calculated?**
> It's not a ratio — it's a minimum-score cutoff. A wave fires only if its competitive score `s(c)` exceeds 1.5.

**Q: How does it capture efficiently?**
> The capture floor: the minimum ships to take a target = ⌈defenders at arrival + overhead + reinforcements⌉. It never sends more than needed, so spare ships go to other targets.

**Q: What's focus-fire?**
> Combining fleets from several planets that arrive on the same turn, so together they clear the floor when no single planet can.

**Q: How does it hit a moving planet?**
> Lead-aiming via an intercept-angle solver (fixed-point iteration), plus a reachability check that accounts for orbital drift and the sun blocking the path.

**Q: Why does it beat your ctx2?**
> It optimises the real simulated outcome instead of imitating a teacher (no imitation ceiling), uses ships efficiently, coordinates strikes, and re-plans every turn so it has no distribution shift.

**Q: Biggest weakness / what would you improve?**
> It doesn't model the opponent — it assumes rivals launch nothing new. The fix is a shallow adversarial roll-out: simulate the strongest rival's best reply and pick moves that survive it.

**Q: Isn't this just the standard public planner everyone used?**
> The techniques — MPC, capture floor, competitive scoring — are standard planning ideas, yes. What we contribute is the side-by-side comparison of planning against a learned re-ranker and the analysis of why planning wins. *(Stay modest on the agent, pivot to the analysis.)*

**Q: Where did the labels / training data come from for NeRL?**
> Trick question — there are none. NeRL has no training and no data. It's a deterministic planner; given a seed, a match reproduces exactly.

### 9.10 NeRL cheat numbers
- Score **1141.8** (beats ctx2 ~1001). Head-to-head: **1211–0** vs heuristic, **1478–0** vs ctx2, both before turn 100.
- Horizon **18 (2p) / 13 (4p)**, **6** waves/turn, ROI cutoff **1.5**, min launch **4** ships, focus-fire up to **4 (2p) / 3 (4p)** sources.
- Scorer: **`s(c) = Δη_me − Σ Δη_others`**. No NN, no weights; `torch` = array math only.

---

*Good luck. Re-read Part 7, Part 8, and the Part 9 Q&A last. Speak plainly, admit what isn't RL, lean on the "learning vs planning" comparison — it's accurate, it's yours, and it's a strong story.*
