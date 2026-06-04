# Orbit Wars AI Experiment Log

## Current Best Configuration
- **Snapshot averaging**: Simple average (v6_1017-style)
- **FWD_SNAPSHOT_TURNS**: (3, 6, 10, 15, 20)
- **SEARCH_MAX_ACTIONS**: 12
- **Target sorting**: (production, ships)
- **Result**: 36.25-53.75% (BEST: 53.75%)

## Improvements Tried

### Successful Improvements
1. **Simple average snapshot averaging (v6_1017-style)**
   - Result: 36.25-53.75% (BEST: 53.75%)
   - Status: **CURRENT BEST**
   - Notes: Only improvement that consistently helps

### Failed Improvements
1. **Phase-aware parameters**
   - Result: REGRESSED
   - Status: REVERTED

2. **Distance-based target selection**
   - Result: CATASTROPHIC (0.00%)
   - Status: REVERTED

3. **Fleet mobility bonus**
   - Result: REGRESSED
   - Status: REVERTED

4. **Late-game flush logic**
   - Result: REGRESSED
   - Status: REVERTED

5. **Depth-2 search**
   - Result: Reverted (API incompatibility)
   - Status: REVERTED

6. **v6_1017 config (full config)**
   - Result: 26.25% (REGRESSION from 38.75%)
   - Status: REVERTED

7. **v6_1017 full model**
   - Result: 27.5% (worse than User)
   - Status: NOT ADOPTED

8. **Terminal Phase logic**
   - Result: 38.75% (no improvement, within variance)
   - Status: REVERTED

9. **FWD_SNAPSHOT_TURNS=(4,8,13,18)**
   - Result: 32.50-33.75% (worse than (3,6,10,15,20))
   - Status: REVERTED

10. **SEARCH_MAX_ACTIONS=8**
    - Result: 26.25-31.25% (REGRESSION from 38.75%)
    - Status: REVERTED

11. **Monte Carlo Tree Search (MCTS)**
    - Result: 10.00% (CATASTROPHIC REGRESSION from 38.75%)
    - Status: REVERTED

12. **Weighted 1/t snapshot averaging (councilHeuristic-style)**
    - Result: 32.50-36.25% (NO IMPROVEMENT, within variance of simple average)
    - Status: REVERTED

13. **Target diversity (F16_DIVERSITY_ENABLED)**
    - Result: 33.75-45.00% (HIGH VARIANCE, BEST 45.00% but unstable)
    - Status: REVERTED

14. **FWD_SNAPSHOT_TURNS=(2,4,6,8,10) - short-term focus**
    - Result: 33.75-45.0% (HIGH VARIANCE, BEST 45.0% but unstable)
    - Status: REVERTED

15. **VAL_PROD_W=10.0 (increased from 8.0)**
    - Result: 36.25-43.75% (HIGH VARIANCE, BEST 43.75% but unstable)
    - Status: REVERTED

16. **LEADER_BASH_ENABLED (councilHeuristic feature)**
    - Result: 36.25-40.0% (NO IMPROVEMENT, within variance of baseline)
    - Status: REVERTED

17. **NEUTRAL_SATURATION_STOP_EXPAND_ENABLED (councilHeuristic feature)**
    - Result: 27.50-30.00% (REGRESSION - worse than baseline)
    - Status: REVERTED

18. **Absolute scoring (completely different approach)**
    - Result: 31.25% (REGRESSION - worse than baseline)
    - Status: REVERTED

19. **FWD_HORIZON=24 (increased from 18)**
    - Result: 31.25-36.25% (REGRESSION/NO IMPROVEMENT)
    - Status: REVERTED

## Improvements NOT YET TRIED (from councilHeuristic)

### High Priority
1. **Mode detection (PERSONALITY_ENABLED)**
   - Status: Code exists but DISABLED
   - Notes: councilHeuristic enabled, User has code but disabled
   - Potential: Adaptive strategy based on enemy aggression

### Medium Priority
2. **fleet_target_planet caching**
   - Status: NOT YET TRIED
   - Notes: Optimization to reduce compute from ~176,000 operations
   - Potential: Performance improvement, may enable deeper search

3. **MULTIPRONG_ENABLED**
   - Status: NOT YET TRIED
   - Notes: Multiprong attacks to force opponent split defense
   - Potential: Strategic advantage

4. **NEUTRAL_SATURATION_STOP_EXPAND_ENABLED**
   - Status: NOT YET TRIED
   - Notes: Stop expansion when neutrals exhausted (2P only)
   - Potential: Better resource allocation

### Low Priority
5. **HAMMER_ENABLED**
   - Status: NOT YET TRIED
   - Notes: Aggressive late-game attacks
   - Potential: Late-game advantage

6. **LEADER_BASH_ENABLED**
   - Status: NOT YET TRIED
   - Notes: Aggressive when leading
   - Potential: Maintain lead

7. **COUNTER_SNIPE_ENABLED**
   - Status: NOT YET TRIED
   - Notes: Counter enemy snipes
   - Potential: Defensive improvement

8. **CHEAP_PICKUP_ENABLED**
   - Status: NOT YET TRIED
   - Notes: Capture weak neutrals
   - Potential: Early-game advantage

## Competition Results History

### Baseline (before improvements)
- User: ~30-40%
- NovaHeuristic: 75-80%

### After simple average snapshot averaging
- Run 1: User 53.75%, NovaHeur 75.00%, v6_1017 23.75%
- Run 2: User 37.5%, NovaHeur 77.5%, v6_1017 35.0%
- Run 3: User 36.25%, NovaHeur 81.25%, v6_1017 30.00%

### After MCTS (CATASTROPHIC)
- User: 10.00%
- NovaHeur: 86.25%
- v6_1017: 51.25%

### After weighted 1/t averaging
- Run 1: User 36.25%, NovaHeur 67.50%, v6_1017 41.25%
- Run 2: User 32.50%, NovaHeur 78.75%, v6_1017 28.75%

### After Target diversity (HIGH VARIANCE)
- Run 1: User 45.00%, NovaHeur 73.75%, v6_1017 30.00%
- Run 2: User 33.75%, NovaHeur 83.75%, v6_1017 38.75%

### After FWD_SNAPSHOT_TURNS=(2,4,6,8,10) (HIGH VARIANCE)
- Run 1: User 45.0%, NovaHeur 75.0%, v6_1017 30.0%
- Run 2: User 33.75%, NovaHeur 76.25%, v6_1017 35.00%

### After VAL_PROD_W=10.0 (HIGH VARIANCE)
- Run 1: User 43.75%, NovaHeur 77.50%, v6_1017 41.25%
- Run 2: User 36.25%, NovaHeur 80.00%, v6_1017 42.50%

### After LEADER_BASH_ENABLED (councilHeuristic feature)
- Run 1: User 36.25%, NovaHeur 75.00%, v6_1017 36.25%
- Run 2: User 40.0%, NovaHeur 72.5%, v6_1017 40.0%

### After NEUTRAL_SATURATION_STOP_EXPAND_ENABLED (REGRESSION)
- Run 1: User 30.00%, NovaHeur 76.25%, v6_1017 38.75%
- Run 2: User 27.50%, NovaHeur 73.75%, v6_1017 42.50%

### After Absolute scoring (REGRESSION)
- Run 1: User 31.25%, NovaHeur 77.50%, v6_1017 45.00%
- Run 2: User 31.25%, NovaHeur 76.25%, v6_1017 36.25%

### After FWD_HORIZON=24 (REGRESSION/NO IMPROVEMENT)
- Run 1: User 31.25%, NovaHeur 78.75%, v6_1017 35.00%
- Run 2: User 36.25%, NovaHeur 78.75%, v6_1017 30.00%

## Conclusion
- **Best achievable with current architecture**: 38.75% (variance 36.25-53.75%)
- **Gap to NovaHeuristic**: ~32.5%
- **Heuristic approach**: EXHAUSTED
- **Next steps**: Try learning-based approaches (RL, Neural Networks, MCTS with proper implementation)

## Notes
- User model consistently beats v6_1017 in competition
- Simple average snapshot averaging is the ONLY improvement that helps
- All parameter tuning approaches have regressed
- Only structural changes (like snapshot averaging) have potential
