# `pokers` fork API 探針結論（Ultimate Texas Hold'em）

探針時間：2026-06-12
Fork: `git+https://github.com/alan19742/pokers.git` @ `ba7e182`

## 探針命令與結果

```bash
python -c "import pokers as pkrs; print([x for x in dir(pkrs) if not x.startswith('_')])"
# ['Action', 'ActionEnum', 'ActionRecord', 'BonusActionEnum', 'BonusState', 'BonusStatus',
#  'Card', 'PlayerState', 'Stage', 'State', 'StateStatus',
#  'parallel_apply_action', 'pokers', 'visualize_state', 'visualize_trace']

python -c "import pokers as pkrs; s=pkrs.BonusState.from_seed(ante=10.0, bonus_bet=1.0, stake=1000.0, seed=0); print([x for x in dir(s) if not x.startswith('_')])"
# ['ante', 'apply_action', 'bonus_bet', 'dealer_hand', 'dealer_revealed', 'deck',
#  'final_state', 'flop_bet', 'from_action', 'from_deck', 'from_seed', 'legal_actions',
#  'player_hand', 'public_cards', 'reward', 'river_bet', 'stage', 'stake', 'status', 'turn_bet']
```

## 關鍵結論

1. **`pkrs.BonusState` 不是 Ultimate Texas Hold'em**，而是 *Texas Hold'em Bonus Poker*
   （見 fork 源碼 `src/bonus.rs` 頭部註釋）：
   - 動作序列：Preflop `Fold | Play(2x ante)` → Flop `Check | Bet(1x)` → Turn `Check | Bet(1x)`；
   - 沒有 Blind 注、沒有莊家資格（pair qualifier）規則、沒有 Trips paytable；
   - 邊注 `bonus_bet` 是基於兩張底牌的 Bonus 賠率表（AA 30:1 等），與 UTH 的 Trips 完全不同。
2. **fork 未向 Python 暴露 7 張牌評估器**（`rank_card_combination` 是 crate 私有函數，
   `lib.rs` 只註冊了 State/BonusState/Card 等類）。
3. `pkrs.Card`：`int(card.suit)` ∈ 0..3、`int(card.rank)` ∈ 0..12（0=2 … 12=A），
   `card_idx = suit * 13 + rank`，與 `src/core/model.py::encode_state` 的編碼一致。
4. `pkrs.State.from_deck(n_players, button, sb, bb, stake, deck)` 可用 —— 用於以
   heads-up 攤牌結果交叉驗證純 Python 評估器（見 `tests/test_uth_env.py`）。

## 因此採取的方案（按任務書應變分支）

- UTH 規則引擎以**純 Python** 實現於 `src/core/uth_env.py`（`UTHState` + `settle_uth`），
  接口與 `pkrs.State` duck-type 兼容（`players_state` / `public_cards` / `stage` /
  `legal_actions` / `apply_action` / `status` / `final_state` / `from_action` …），
  以便直接復用 `src/core/model.py::encode_state`。
- 牌型評估器（7 選 5）在 `uth_env.py` 內實現，並在
  `tests/test_uth_env.py::test_evaluator_matches_pokers_on_random_samples` 中以 `pkrs.State.from_deck`
  heads-up 攤牌的 reward 符號做 100% 一致性交叉驗證。
- `BonusState` 保持不動，舊的 NLHE / Bonus 路徑不受影響。

## 交叉驗證中發現的 `pokers` 引擎 bug

- **Wheel（A-2-3-4-5）誤判**：Rust 引擎把 A 作低的順子當成 A 高順子排名，
  導致「7 高順子 vs wheel」被判為平局（實測 trial 98：`6h7d` vs `Td2c`，
  board `5c 2h Ac 4c 3s`）。純 Python 評估器的處理是正確的（wheel 高牌為 5）。
  交叉驗證測試對 wheel 案例做了跳過處理，並用
  `test_wheel_straight_ordering` 單獨鎖定正確行為。
- **既有失敗（與 UTH 無關）**：`tests/test_pokers_regressions.py` 中
  `all-in-runout-goes-straight-to-showdown` 與 `no-raise-when-call-uses-entire-stack`
  兩條在 fork 的 `main` 與 `v0/huzaifaansari87654-…` 分支構建下均失敗 ——
  readme 所述的 all-in 補丁不在當前任何遠端分支中。本次 UTH 工作未觸及該路徑。

## UTH 賠率表（BGC 官方，2015-03 修訂版）

Blind（四套表相同）：RF 500:1、SF 50:1、4K 10:1、FH 3:1、Flush 3:2、Straight 1:1，低於順子 push。

| Trips Bonus | UTH-01 | UTH-02 | UTH-03 | UTH-04 |
|---|---|---|---|---|
| Royal Flush | 50 | 50 | 50 | 50 |
| Straight Flush | 40 | 40 | 40 | 40 |
| Four of a Kind | 30 | 30 | 30 | 20 |
| Full House | 9 | 8 | 8 | 7 |
| Flush | 7 | 6 | 7 | 6 |
| Straight | 4 | 5 | 4 | 5 |
| Three of a Kind | 3 | 3 | 3 | 3 |
