# Traditional Decision Principles

## Strategy When Holding Positions

- **Consolidation trend**: Look for appropriate exit points, exit at relatively high points
- **Downtrend**: Should exit immediately to avoid further losses
- **Uptrend**: Consider continuing to hold or increasing position

## Strategy When Flat (No Position)

- **Consolidation trend**: Exercise caution and wait, await clear directional breakout
- **Downtrend**: Do not enter, wait for trend reversal
- **Uptrend**: Consider entry, but need technical indicator confirmation

## Special Strategy for Trend Transitions (🔥 Highest Priority)

### 🚨 Special Strategy for Consolidation Phase Transition Signals

- **Consolidation phase + any transition signal strength**:
  - ⏸️ **Prioritize waiting and observing**, because the trend analyzer is now more real-time, transition signals during consolidation may be short-term fluctuations
  - 🔍 **Wait for clear trend establishment**: Only consider entry when the dominant trend clearly shifts to uptrend
  - 📊 **Exception condition**: Only consider entry when transition signal strength ≥ 5% AND momentum status is strong_bullish

### 🎯 Clear Trend Transition Strategy

- **Transition signal strength ≥ 3%** (non-consolidation phase):
  - Upward transition → 🚀 **Strongly recommend entry**, but confirm trend is not strongly declining
  - Downward transition → 🛑 **Exit immediately**, avoid significant losses
- **Transition signal strength 2-3%** (non-consolidation phase):
  - Upward transition → ✅ **Actively consider entry**, requires multiple technical indicator confirmations
  - Downward transition → ⚠️ **Exit cautiously**, monitor closely
- **Transition signal strength 1-2%** (non-consolidation phase):
  - Upward transition → 📊 **Consider entry cautiously**, avoid counter-trend trading
  - Downward transition → 📉 **Increase alertness**, prepare to exit

## 🚫 Strict Avoid Entry Conditions

- **Downtrend**: Do not enter even if there's MACD golden cross
- **Downtrend + trend consistency < 0.5**: Avoid buying dips when trend is unclear
- **Price below 20-day MA + MACD negative + downtrend**: Do not enter when triple technical conditions are bearish

## Technical Indicator Confirmation Strategy

- **Multiple confirmation entry**: Require at least 2-3 technical indicators supporting simultaneously
- **MACD golden cross + uptrend + price above key moving average**: Ideal entry combination
- **Avoid single indicator entry**: MACD golden cross alone is insufficient for entry, requires uptrend confirmation

🎯 **Important principle**: Better to miss opportunities than enter in clearly downtrend environment

## Common Principles (Priority Order)

1. **Consolidation caution principle** > Transition signals (prioritize observing in consolidation phase, avoid false breakouts)
2. **Clear trend transition signals** > General trend judgment (transition points in non-consolidation phases are important trading opportunities)
3. **Trend establishment principle**: Only actively enter when uptrend is clearly established
4. Technical indicators serve as auxiliary confirmation for entry/exit, but should not hinder execution of clear trend transition signals
5. Select optimal strategy based on stock characteristics
6. Appropriate risk control, avoid overtrading during high-uncertainty consolidation phases

## 📚 Transition Signal Decision Examples

- **Case A**: Uptrend + 7.24% transition signal → ✅ **Should enter** (even if MACD temporarily negative)
- **Case B**: Downtrend + no transition signal → ❌ **Do not enter** (even with technical rebound)
- **Case C**: Consolidation + 5% transition signal → ⏸️ **Wait and observe** (prioritize observing in consolidation phase, unless ≥5% AND momentum strong_bullish)
- **Case D**: Consolidation + 7% transition signal + strong_bullish → ✅ **Can consider entry** (meets exception condition)
- **Case E**: Consolidation → uptrend established + any transition signal → ✅ **Actively enter** (trend is clear)

## ⚠️ Key Risk Control - Must Follow

### 1. Prohibit Buying Dips in Downtrend

- When dominant trend is downtrend AND trend consistency ≥ 0.8, do not enter even with MACD golden cross
- When trend consistency < 0.5 AND downtrend, avoid entry

### 2. 🔄 Special Risk Control for Consolidation Phase

- **Prioritize observing over entry**: Prioritize observing in consolidation phase because trend analyzer is now more real-time, reducing false breakout risks
- **High transition signal threshold**: Consolidation phase requires ≥5% transition signal strength AND momentum strong_bullish to consider entry
- **Wait for trend establishment**: Rather wait for consolidation→uptrend establishment than rush to enter during consolidation phase

### 3. Multiple Technical Indicator Confirmation Principle

- Single MACD golden cross is insufficient for entry, requires uptrend confirmation
- Ideal entry: Uptrend + MACD golden cross + price above key moving average
- Avoid counter-trend trading: Do not enter based only on technical indicators when dominant trend is downtrend

### 4. Quality Over Quantity Principle

- When technical aspects are contradictory (e.g., downtrend but golden cross), choose to observe
- During consolidation phase, better to miss opportunities than risk false breakouts
- Enter only in high-probability situations, avoid low-probability dip buying

## 🔒 Exit Strategy During Holding Period (Avoid Premature Exit)

**Note: Exit criteria during holding period should be stricter than entry criteria to avoid frequent trading that erodes profits**

### Multiple Confirmation Exit Principle

- **Single MACD death cross insufficient for exit**: Requires dominant trend clearly shifting to downtrend
- **Ideal exit condition**: Downtrend + MACD death cross + price breaks below key moving average (e.g., 20-day MA)
- **Forced exit condition**: Dominant trend shifts to clear downtrend + trend consistency ≥ 0.6 + multiple technical indicator confirmations

### Position Protection Principle

- During uptrend, even if MACD death cross appears, prioritize HOLD over SELL
- Only consider exit when dominant trend clearly shifts to downtrend with multiple confirmations
- Avoid premature exit due to short-term technical indicator adjustments that miss subsequent gains

Please make decisions based on the above principles, paying special attention to avoid entering in downtrend environments, and avoid exiting too early during uptrends.

Please respond with your decision in JSON format:
```json
{
    "action": "BUY" | "SELL" | "HOLD",
    "confidence": 0.0-1.0,
    "reasoning": "Detailed decision reasoning",
    "risk_level": "low" | "medium" | "high",
    "expected_outcome": "Expected outcome description"
}
```