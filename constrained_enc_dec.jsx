import { useState, useEffect, useRef, useMemo } from "react";

// ─── DESIGN TOKENS ───
const C = {
  bg: "#06090f",
  surface: "#0c1018",
  raised: "#121a28",
  hover: "#182234",
  border: "#1c2940",
  borderActive: "#2a3d5c",
  text: "#d8e2f0",
  muted: "#7e90a8",
  dim: "#4a5c72",
  // Semantic
  sell: "#f472b6",     // pink - sell price
  sellDim: "rgba(244,114,182,0.12)",
  contract: "#38bdf8", // sky - contract
  contractDim: "rgba(56,189,248,0.12)",
  constraint: "#fb923c", // orange - constraint/floor
  constraintDim: "rgba(251,146,60,0.12)",
  encoder: "#a78bfa",  // violet - encoder
  encoderDim: "rgba(167,139,250,0.12)",
  decoder: "#34d399",  // emerald - decoder
  decoderDim: "rgba(52,211,153,0.12)",
  amber: "#fbbf24",
  amberDim: "rgba(251,191,36,0.12)",
  red: "#f87171",
  redDim: "rgba(248,113,113,0.12)",
  white: "#ffffff",
  alpha: "#facc15",    // yellow for alpha
};

const font = "'Source Code Pro', 'IBM Plex Mono', monospace";
const fontSans = "'Sora', 'DM Sans', sans-serif";

// ─── DATA GENERATION ───
function seeded(seed) {
  let s = seed;
  return () => { s = (s * 16807) % 2147483647; return (s - 1) / 2147483646; };
}

function generateContractSeries(n = 24) {
  const r = seeded(42);
  let base = 100;
  return Array.from({ length: n }, (_, i) => {
    if (i % 6 === 0 && i > 0) base += 8 + r() * 12; // step increases at contract renewals
    const v = base + (r() - 0.5) * 3;
    return parseFloat(v.toFixed(2));
  });
}

function generateSellSeries(contract, alpha = 3) {
  const r = seeded(99);
  return contract.map((_, i) => {
    // Sell price follows contract with alpha delay + margin + noise
    const refIdx = Math.max(0, i - alpha);
    const refContract = contract[refIdx];
    const margin = 5 + r() * 15; // 5-20% margin above contract
    const noise = (r() - 0.5) * 8;
    let sell = refContract + margin + noise;
    // Sometimes sell dips below current contract (the problem!)
    if (i > 0 && i % 7 === 0) sell = contract[i] - 2 - r() * 5; // violation!
    return parseFloat(sell.toFixed(2));
  });
}

// ─── REUSABLE COMPONENTS ───
function Pill({ children, active, onClick, color = C.contract }) {
  return (
    <button onClick={onClick} style={{
      padding: "6px 16px", borderRadius: 20,
      border: `1.5px solid ${active ? color : C.border}`,
      background: active ? color + "18" : "transparent",
      color: active ? color : C.muted,
      fontSize: 12, fontWeight: 600, cursor: "pointer",
      fontFamily: fontSans, transition: "all 0.15s", whiteSpace: "nowrap",
    }}>{children}</button>
  );
}

function Badge({ children, color }) {
  return (
    <span style={{
      display: "inline-block", padding: "2px 8px", borderRadius: 10,
      background: color + "20", color, fontSize: 10, fontWeight: 700,
      fontFamily: font, letterSpacing: "0.3px",
    }}>{children}</span>
  );
}

function Card({ children, style = {}, glow }) {
  return (
    <div style={{
      background: C.surface, borderRadius: 14,
      border: `1px solid ${C.border}`, padding: 20,
      ...(glow ? { boxShadow: `0 0 24px ${glow}` } : {}),
      ...style,
    }}>{children}</div>
  );
}

function Title({ children, icon, color = C.text }) {
  return (
    <h3 style={{
      fontSize: 15, fontWeight: 700, color, margin: "0 0 12px",
      display: "flex", alignItems: "center", gap: 8, fontFamily: fontSans,
    }}>{icon && <span style={{ fontSize: 16 }}>{icon}</span>}{children}</h3>
  );
}

// ─── KERAS ARCHITECTURE PANE ───
function KerasArchPane({ alpha }) {
  const [hovered, setHovered] = useState(null);

  const encoderLayers = [
    {
      id: "enc_input", name: "Encoder Input", code: `Input(shape=(T, 2), name='encoder_input')`,
      shape: "(None, T, 2)", params: 0, color: C.contract,
      desc: `Receives T timesteps × 2 features: [sell_price, contract_value]. Both series are fed jointly so the encoder can learn their relationship and the α=${alpha} lag pattern.`,
      detail: `# T = lookback window (e.g., 24 months)\n# 2 features: sell_price + contract_value\n# Shape: (batch_size, T, 2)`,
    },
    {
      id: "enc_lstm1", name: "Encoder LSTM 1", code: `LSTM(128, return_sequences=True, name='enc_lstm_1')`,
      shape: "(None, T, 128)", params: ((2 + 128) * 128 * 4 + 128 * 4),
      color: C.encoder,
      desc: "First encoder LSTM. return_sequences=True so the second layer can see every timestep's representation. Learns temporal patterns: trend, seasonality, and the α-lag relationship.",
      detail: `# Params: 4 gates × (input_dim + units) × units + biases\n# = 4 × (2 + 128) × 128 + 4 × 128\n# This layer learns HOW sell follows contract`,
    },
    {
      id: "enc_lstm2", name: "Encoder LSTM 2", code: `LSTM(64, return_sequences=False, return_state=True, name='enc_lstm_2')`,
      shape: "(None, 64) + states", params: ((128 + 64) * 64 * 4 + 64 * 4),
      color: C.encoder,
      desc: "Second encoder LSTM. return_state=True is CRITICAL — it outputs [output, hidden_state, cell_state]. These states become the decoder's initial condition, carrying all encoded knowledge.",
      detail: `# return_state=True → returns 3 tensors:\n#   output:       (batch, 64)  — last output\n#   hidden_state: (batch, 64)  — h(T) \n#   cell_state:   (batch, 64)  — c(T)\n# These are the "context" for the decoder`,
    },
  ];

  const decoderLayers = [
    {
      id: "dec_input", name: "Decoder Input", code: `Input(shape=(H, 1), name='decoder_input')`,
      shape: "(None, H, 1)", params: 0, color: C.decoder,
      desc: `Receives the KNOWN future contract values for the forecast horizon H. This is the key: contracts are signed ahead of time, so we KNOW them. The decoder uses this as "scheduled" input.`,
      detail: `# H = forecast horizon (e.g., 6 months ahead)\n# 1 feature: future contract values (KNOWN)\n# This is what makes it "teacher forcing"\n# with real future information`,
    },
    {
      id: "dec_lstm", name: "Decoder LSTM", code: `LSTM(64, return_sequences=True, name='dec_lstm')`,
      shape: "(None, H, 64)", params: ((1 + 64) * 64 * 4 + 64 * 4),
      color: C.decoder,
      desc: "Decoder LSTM initialized with encoder's [h(T), c(T)]. At each future step, it receives the known contract value and generates a hidden state that incorporates BOTH the encoded history AND the future contract.",
      detail: `# initial_state = [enc_hidden, enc_cell]\n# This is the "bridge" — the decoder starts\n# with full knowledge of the past encoded\n# by the encoder, then sees future contracts`,
    },
  ];

  const constraintLayers = [
    {
      id: "dense_raw", name: "Dense (Raw Pred)", code: `Dense(1, activation='linear', name='raw_prediction')`,
      shape: "(None, H, 1)", params: 64 + 1,
      color: C.sell,
      desc: "Produces the unconstrained raw sell price prediction from the decoder's hidden states. This prediction may violate the contract floor.",
      detail: `# raw_pred(t) = W × h_dec(t) + b\n# This is the model's "best guess" before\n# any constraint is applied`,
    },
    {
      id: "alpha_shift", name: "Alpha-Shift Layer", code: `Lambda(shift_contract_by_alpha, name='alpha_shift')`,
      shape: "(None, H, 1)", params: 0, color: C.alpha,
      desc: `Custom layer that shifts the contract input by α=${alpha} timesteps. Since sell prices are delayed by α steps relative to contract changes, the effective floor at time t is contract(t−α), not contract(t).`,
      detail: `# def shift_contract_by_alpha(contract_seq):\n#     # Pad start with first value, trim end\n#     shifted = tf.concat([\n#         tf.repeat(contract_seq[:, :1, :], ${alpha}, axis=1),\n#         contract_seq[:, :-${alpha}, :]\n#     ], axis=1)\n#     return shifted\n# Floor at time t = contract(t - alpha)`,
    },
    {
      id: "constraint", name: "Constraint Layer", code: `Lambda(apply_floor_constraint, name='constrained_output')`,
      shape: "(None, H, 1)", params: 0, color: C.constraint,
      desc: "The CORE constraint: final_pred(t) = max(raw_pred(t), shifted_contract(t) + margin). Ensures sell price never falls below the α-shifted contract floor. Uses a soft-max (LogSumExp) during training for differentiability.",
      detail: `# HARD constraint (inference):\n# final = tf.maximum(raw_pred, floor + margin)\n#\n# SOFT constraint (training — differentiable):\n# softmax(a,b) = log(exp(a) + exp(b))\n# final = softmax(raw_pred, floor + margin)\n#\n# margin = learned or fixed buffer above contract`,
    },
  ];

  const allLayers = [
    { header: "ENCODER — Reads History", layers: encoderLayers, color: C.encoder },
    { header: "DECODER — Generates Future", layers: decoderLayers, color: C.decoder },
    { header: "CONSTRAINT HEAD — Enforces Floor", layers: constraintLayers, color: C.constraint },
  ];

  const totalParams = [...encoderLayers, ...decoderLayers, ...constraintLayers].reduce((a, l) => a + l.params, 0);

  const fullCode = `import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Lambda, Concatenate
)

# ═══════════════════════════════════════════════
# Constrained Encoder-Decoder LSTM
# ═══════════════════════════════════════════════
# Series 1: Historical sell prices
# Series 2: Historical contract values  
# Constraint: sell ≥ contract(t - α) + margin
# α = ${alpha} (contract-to-sell lag in timesteps)
# ═══════════════════════════════════════════════

ALPHA = ${alpha}          # Lag: contract takes α steps to affect sell
T = 24              # Encoder lookback window  
H = 6               # Decoder forecast horizon
MARGIN = 2.0        # Minimum margin above contract floor
SOFTMAX_TEMP = 5.0  # Sharpness of soft constraint

# ─── Alpha-shift: effective contract floor ───
def shift_contract_by_alpha(contract_seq):
    """
    The contract at time t doesn't affect sell
    price until time t + alpha. So the effective
    floor at forecast step t is:
        floor(t) = contract(t - alpha)
    
    We shift the known future contract sequence
    backward by alpha steps.
    """
    padding = tf.repeat(
        contract_seq[:, :1, :], ALPHA, axis=1
    )
    shifted = tf.concat(
        [padding, contract_seq[:, :-ALPHA, :]],
        axis=1
    )
    return shifted

# ─── Soft floor constraint (differentiable) ───
def apply_floor_constraint(inputs):
    """
    Ensures: sell_pred >= floor + margin
    
    During training, uses smooth approximation:
      softmax(a, b) = log(exp(τa) + exp(τb)) / τ
    
    This is differentiable everywhere, so gradients
    flow through the constraint. As τ → ∞, it 
    approaches hard max().
    """
    raw_pred, shifted_contract = inputs
    floor = shifted_contract + MARGIN
    
    # LogSumExp smooth maximum
    stacked = tf.stack(
        [raw_pred * SOFTMAX_TEMP,
         floor * SOFTMAX_TEMP], axis=-1
    )
    constrained = (
        tf.reduce_logsumexp(stacked, axis=-1,
                            keepdims=True)
        / SOFTMAX_TEMP
    )
    return constrained

# ═══════════════════════════════════════════════
# MODEL DEFINITION
# ═══════════════════════════════════════════════

# ─── ENCODER ───
encoder_input = Input(
    shape=(T, 2), name='encoder_input'
)
# Input features: [sell_price, contract_value]

enc_x = LSTM(
    128, return_sequences=True,
    name='enc_lstm_1'
)(encoder_input)

# return_state=True → we get h(T) and c(T)
enc_out, enc_h, enc_c = LSTM(
    64, return_sequences=False,
    return_state=True,
    name='enc_lstm_2'
)(enc_x)

# enc_h, enc_c carry ALL encoded knowledge
# about the sell-contract relationship,
# the α-lag pattern, and temporal dynamics

# ─── DECODER ───
decoder_input = Input(
    shape=(H, 1), name='decoder_input'
)
# Known future contract values (H steps ahead)

dec_x = LSTM(
    64, return_sequences=True,
    name='dec_lstm'
)(decoder_input, initial_state=[enc_h, enc_c])
#                ^^^^^^^^^^^^^^^^^^^^^^^^^^
# CRITICAL: decoder starts from encoder's
# final states — this is the "context bridge"

# ─── RAW PREDICTION ───
raw_pred = Dense(
    1, activation='linear',
    name='raw_prediction'
)(dec_x)
# Shape: (batch, H, 1) — unconstrained sell pred

# ─── ALPHA-SHIFTED CONTRACT FLOOR ───
shifted_contract = Lambda(
    shift_contract_by_alpha,
    name='alpha_shift'
)(decoder_input)
# Shape: (batch, H, 1) — floor(t) = contract(t-α)

# ─── APPLY CONSTRAINT ───
constrained_pred = Lambda(
    apply_floor_constraint,
    name='constrained_output'
)([raw_pred, shifted_contract])
# Shape: (batch, H, 1) — final sell price forecast
# Guarantee: pred(t) >= contract(t-α) + margin

# ─── ASSEMBLE MODEL ───
model = Model(
    inputs=[encoder_input, decoder_input],
    outputs=constrained_pred,
    name='constrained_enc_dec_lstm'
)

# ─── CUSTOM LOSS ───
def constrained_loss(y_true, y_pred):
    """
    Composite loss:
    1. MSE for accuracy
    2. Penalty for violating constraint
    """
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # Extra penalty if prediction is below floor
    # (shouldn't happen with constraint layer,
    #  but adds gradient pressure during training)
    violation = tf.maximum(0.0, floor_value - y_pred)
    penalty = tf.reduce_mean(tf.square(violation))
    
    return mse + 10.0 * penalty

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3
    ),
    loss='mse',  # or constrained_loss
    metrics=['mae']
)

model.summary()
# Total params: ~${totalParams.toLocaleString()}

# ═══════════════════════════════════════════════
# TRAINING DATA PREPARATION
# ═══════════════════════════════════════════════
# 
# X_encoder: (samples, T, 2)
#   - [:, :, 0] = historical sell prices
#   - [:, :, 1] = historical contract values
#
# X_decoder: (samples, H, 1)  
#   - [:, :, 0] = KNOWN future contract values
#   - (contracts are signed ahead of time!)
#
# Y: (samples, H, 1)
#   - [:, :, 0] = actual future sell prices
#
# model.fit(
#     [X_encoder, X_decoder], Y,
#     epochs=100, batch_size=32,
#     validation_split=0.2
# )`;

  return (
    <div>
      {allLayers.map((section, si) => (
        <div key={si} style={{ marginBottom: 20 }}>
          <div style={{
            fontSize: 11, fontWeight: 700, color: section.color,
            fontFamily: font, letterSpacing: "1px", marginBottom: 8,
            textTransform: "uppercase", paddingLeft: 4,
          }}>
            {section.header}
          </div>

          {section.layers.map((layer, li) => {
            const isH = hovered === layer.id;
            return (
              <div key={layer.id} style={{ marginBottom: 4 }}>
                <div
                  onMouseEnter={() => setHovered(layer.id)}
                  onMouseLeave={() => setHovered(null)}
                  onClick={() => setHovered(isH ? null : layer.id)}
                  style={{
                    display: "grid",
                    gridTemplateColumns: "155px 1fr 100px 70px",
                    alignItems: "center", gap: 10,
                    padding: "9px 12px", borderRadius: 8,
                    border: `1.5px solid ${isH ? layer.color : C.border}`,
                    background: isH ? layer.color + "0a" : C.raised,
                    cursor: "pointer", transition: "all 0.15s",
                  }}
                >
                  <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
                    <div style={{ width: 7, height: 7, borderRadius: "50%", background: layer.color, boxShadow: `0 0 6px ${layer.color}50` }} />
                    <span style={{ fontSize: 12, fontWeight: 600, color: layer.color, fontFamily: font }}>{layer.name}</span>
                  </div>
                  <code style={{ fontSize: 10, color: C.muted, fontFamily: font, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {layer.code}
                  </code>
                  <span style={{ fontSize: 9, color: C.dim, fontFamily: font }}>{layer.shape}</span>
                  <span style={{ fontSize: 9, color: C.dim, fontFamily: font, textAlign: "right" }}>
                    {layer.params > 0 ? layer.params.toLocaleString() : "—"}
                  </span>
                </div>

                {isH && (
                  <div style={{
                    margin: "3px 0 3px 18px", padding: "10px 14px",
                    borderLeft: `2px solid ${layer.color}40`,
                    background: C.surface, borderRadius: "0 8px 8px 0",
                  }}>
                    <p style={{ fontSize: 12, color: C.muted, margin: "0 0 6px", lineHeight: 1.6 }}>{layer.desc}</p>
                    <pre style={{ fontSize: 10, color: layer.color, margin: 0, fontFamily: font, opacity: 0.8, whiteSpace: "pre-wrap" }}>{layer.detail}</pre>
                  </div>
                )}

                {/* Connector */}
                {li < section.layers.length - 1 && (
                  <div style={{ display: "flex", justifyContent: "center", height: 6 }}>
                    <div style={{ width: 1.5, height: 6, background: C.border }} />
                  </div>
                )}
              </div>
            );
          })}

          {/* Section connector */}
          {si < allLayers.length - 1 && (
            <div style={{ display: "flex", justifyContent: "center", padding: "4px 0" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <div style={{ width: 20, height: 1.5, background: C.border }} />
                <span style={{ fontSize: 10, color: C.dim, fontFamily: font }}>
                  {si === 0 ? "[h(T), c(T)] →" : "constraint →"}
                </span>
                <div style={{ width: 20, height: 1.5, background: C.border }} />
              </div>
            </div>
          )}
        </div>
      ))}

      {/* Summary */}
      <div style={{
        display: "flex", justifyContent: "space-between", alignItems: "center",
        padding: "10px 14px", borderRadius: 8,
        background: C.raised, border: `1px solid ${C.border}`, marginBottom: 14,
      }}>
        <span style={{ fontSize: 11, color: C.muted }}>Total trainable parameters</span>
        <span style={{ fontSize: 14, fontWeight: 700, color: C.amber, fontFamily: font }}>{totalParams.toLocaleString()}</span>
      </div>

      {/* Full code */}
      <details>
        <summary style={{ fontSize: 12, fontWeight: 600, color: C.contract, cursor: "pointer", padding: "8px 0", userSelect: "none" }}>
          📋 Full Keras Code — Click to expand
        </summary>
        <pre style={{
          background: C.raised, border: `1px solid ${C.border}`,
          borderRadius: 10, padding: 16, fontSize: 10.5,
          fontFamily: font, color: C.muted, overflow: "auto",
          maxHeight: 500, lineHeight: 1.6, whiteSpace: "pre-wrap",
        }}>{fullCode}</pre>
      </details>
    </div>
  );
}

// ─── MAIN COMPONENT ───
export default function ConstrainedForecast() {
  const [tab, setTab] = useState("problem");
  const [alpha, setAlpha] = useState(3);
  const [activeStep, setActiveStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const intervalRef = useRef(null);

  const SEQ_LEN = 24;
  const FORECAST_H = 6;

  const contractData = useMemo(() => generateContractSeries(SEQ_LEN + FORECAST_H), []);
  const sellData = useMemo(() => generateSellSeries(contractData.slice(0, SEQ_LEN), alpha), [contractData, alpha]);

  // Simulated forecast
  const forecast = useMemo(() => {
    const r = seeded(777);
    return Array.from({ length: FORECAST_H }, (_, i) => {
      const futureContract = contractData[SEQ_LEN + i];
      const shiftedContract = contractData[Math.max(0, SEQ_LEN + i - alpha)];
      const rawPred = shiftedContract + 3 + r() * 18 + (r() > 0.7 ? -12 : 0);
      const floor = shiftedContract + 2; // margin=2
      const constrained = Math.max(rawPred, floor);
      return {
        step: SEQ_LEN + i,
        contract: futureContract,
        shiftedContract,
        floor,
        rawPred: parseFloat(rawPred.toFixed(2)),
        constrained: parseFloat(constrained.toFixed(2)),
        wasViolation: rawPred < floor,
      };
    });
  }, [contractData, alpha]);

  // Detect historical violations
  const violations = useMemo(() => {
    return sellData.map((sell, i) => {
      const effectiveContract = contractData[Math.max(0, i)];
      return sell < effectiveContract;
    });
  }, [sellData, contractData]);

  useEffect(() => {
    if (isPlaying) {
      intervalRef.current = setInterval(() => {
        setActiveStep(s => {
          if (s >= SEQ_LEN - 1) { setIsPlaying(false); return s; }
          return s + 1;
        });
      }, 600);
    }
    return () => clearInterval(intervalRef.current);
  }, [isPlaying]);

  // Chart helpers
  const W = 730, H = 220, pL = 50, pR = 16, pT = 24, pB = 32;
  const plotW = W - pL - pR, plotH = H - pT - pB;
  const allVals = [...sellData, ...contractData, ...forecast.map(f => f.constrained), ...forecast.map(f => f.rawPred)];
  const minV = Math.min(...allVals) - 5, maxV = Math.max(...allVals) + 5;
  const totalLen = SEQ_LEN + FORECAST_H;
  const xS = i => pL + (i / (totalLen - 1)) * plotW;
  const yS = v => pT + plotH - ((v - minV) / (maxV - minV)) * plotH;

  const tabs = [
    { id: "problem", label: "⚡ The Problem" },
    { id: "encoder", label: "🔮 Encoder" },
    { id: "decoder", label: "🎯 Decoder" },
    { id: "constraint", label: "🔒 Constraint" },
    { id: "arch", label: "⚙️ Keras Architecture" },
    { id: "forecast", label: "📈 Full Forecast" },
  ];

  return (
    <div style={{ background: C.bg, minHeight: "100vh", color: C.text, fontFamily: fontSans, padding: "20px 12px" }}>
      <link href="https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600;700&family=Source+Code+Pro:wght@400;500;600;700&display=swap" rel="stylesheet" />

      <div style={{ maxWidth: 790, margin: "0 auto" }}>
        {/* Header */}
        <div style={{ marginBottom: 18 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <div style={{
              width: 38, height: 38, borderRadius: 10,
              background: `linear-gradient(135deg, ${C.sell}30, ${C.contract}30)`,
              display: "flex", alignItems: "center", justifyContent: "center", fontSize: 18,
            }}>📊</div>
            <div>
              <h1 style={{
                fontSize: 21, fontWeight: 700, margin: 0, letterSpacing: "-0.5px",
                background: `linear-gradient(135deg, ${C.sell}, ${C.contract})`,
                WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
              }}>Constrained Encoder-Decoder LSTM</h1>
              <p style={{ fontSize: 11, color: C.dim, margin: "2px 0 0" }}>
                Sell price forecasting · Contract floor constraint · α-lag alignment
              </p>
            </div>
          </div>
        </div>

        {/* Alpha control */}
        <Card style={{ marginBottom: 14, padding: "12px 20px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 16, flexWrap: "wrap" }}>
            <span style={{ fontSize: 12, fontWeight: 600, color: C.alpha, fontFamily: font }}>α = {alpha} timesteps</span>
            <input type="range" min={1} max={6} value={alpha} onChange={e => setAlpha(Number(e.target.value))}
              style={{ width: 140, accentColor: C.alpha }} />
            <span style={{ fontSize: 11, color: C.muted }}>
              Contract changes take <strong style={{ color: C.alpha }}>{alpha} periods</strong> to propagate into sell prices
            </span>
          </div>
        </Card>

        {/* Tabs */}
        <div style={{ display: "flex", gap: 5, marginBottom: 16, flexWrap: "wrap" }}>
          {tabs.map(t => <Pill key={t.id} active={tab === t.id} onClick={() => setTab(t.id)} color={
            t.id === "problem" ? C.red : t.id === "encoder" ? C.encoder : t.id === "decoder" ? C.decoder :
            t.id === "constraint" ? C.constraint : t.id === "arch" ? C.contract : C.sell
          }>{t.label}</Pill>)}
        </div>

        {/* ═══════ PROBLEM TAB ═══════ */}
        {tab === "problem" && (<>
          <Card style={{ marginBottom: 14 }}>
            <Title icon="⚡" color={C.red}>The Problem: Constraint Violations</Title>
            <p style={{ fontSize: 12, color: C.muted, margin: "0 0 14px", lineHeight: 1.7 }}>
              The <span style={{ color: C.sell, fontWeight: 600 }}>sell price</span> should always stay above the <span style={{ color: C.contract, fontWeight: 600 }}>contract value</span> — but it doesn't react immediately. Contract changes take <Badge color={C.alpha}>α = {alpha} steps</Badge> to propagate. During this lag, violations occur (red zones below).
            </p>

            <svg width={W} height={H + 10} viewBox={`0 0 ${W} ${H + 10}`} style={{ width: "100%", height: "auto" }}>
              {/* Grid */}
              {[0,1,2,3,4].map(i => {
                const v = minV + ((maxV - minV) * i) / 4;
                return (<g key={i}>
                  <line x1={pL} x2={W - pR} y1={yS(v)} y2={yS(v)} stroke={C.border} strokeWidth={0.5} />
                  <text x={pL - 6} y={yS(v) + 3} textAnchor="end" fill={C.dim} fontSize={9} fontFamily={font}>${v.toFixed(0)}</text>
                </g>);
              })}

              {/* Violation zones */}
              {sellData.map((sell, i) => {
                if (violations[i]) {
                  return <rect key={`v${i}`} x={xS(i) - (plotW / totalLen) / 2} y={pT} width={plotW / totalLen} height={plotH}
                    fill={C.red} opacity={0.08} />;
                }
                return null;
              })}

              {/* Contract line */}
              <path d={contractData.slice(0, SEQ_LEN).map((v, i) => `${i === 0 ? "M" : "L"}${xS(i)},${yS(v)}`).join(" ")}
                fill="none" stroke={C.contract} strokeWidth={2} opacity={0.8} strokeLinejoin="round" />

              {/* Alpha-shifted contract (effective floor) */}
              <path d={contractData.slice(0, SEQ_LEN).map((_, i) => {
                const shifted = contractData[Math.max(0, i - alpha)];
                return `${i === 0 ? "M" : "L"}${xS(i)},${yS(shifted)}`;
              }).join(" ")}
                fill="none" stroke={C.alpha} strokeWidth={1.5} strokeDasharray="4 3" opacity={0.5} />

              {/* Sell line */}
              <path d={sellData.map((v, i) => `${i === 0 ? "M" : "L"}${xS(i)},${yS(v)}`).join(" ")}
                fill="none" stroke={C.sell} strokeWidth={2} opacity={0.85} strokeLinejoin="round" />

              {/* Violation markers */}
              {sellData.map((sell, i) => violations[i] ? (
                <g key={`vm${i}`}>
                  <circle cx={xS(i)} cy={yS(sell)} r={5} fill={C.red} opacity={0.9} style={{ filter: `drop-shadow(0 0 6px ${C.red})` }} />
                  <text x={xS(i)} y={yS(sell) - 10} textAnchor="middle" fill={C.red} fontSize={8} fontWeight={700} fontFamily={font}>!</text>
                </g>
              ) : null)}

              {/* X labels */}
              {Array.from({ length: SEQ_LEN }, (_, i) => (
                <text key={i} x={xS(i)} y={H + 4} textAnchor="middle" fill={violations[i] ? C.red : C.dim} fontSize={7} fontFamily={font}>
                  M{i + 1}
                </text>
              ))}

              {/* Legend */}
              <g transform="translate(60, 10)">
                <line x1={0} x2={20} y1={0} y2={0} stroke={C.sell} strokeWidth={2} />
                <text x={24} y={4} fill={C.sell} fontSize={9} fontFamily={font}>Sell Price</text>
                <line x1={110} x2={130} y1={0} y2={0} stroke={C.contract} strokeWidth={2} />
                <text x={134} y={4} fill={C.contract} fontSize={9} fontFamily={font}>Contract</text>
                <line x1={210} x2={230} y1={0} y2={0} stroke={C.alpha} strokeWidth={1.5} strokeDasharray="4 3" />
                <text x={234} y={4} fill={C.alpha} fontSize={9} fontFamily={font}>Floor (α-shifted)</text>
                <circle cx={380} cy={0} r={4} fill={C.red} />
                <text x={388} y={4} fill={C.red} fontSize={9} fontFamily={font}>Violation</text>
              </g>
            </svg>
          </Card>

          {/* The alpha lag explanation */}
          <Card>
            <Title icon="⏱️" color={C.alpha}>The α-Lag Mechanism</Title>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10 }}>
              {[
                { n: "1", title: "Contract Signed", desc: "A new manufacturing contract is signed at time t with value C(t). This sets a new cost floor.", color: C.contract, icon: "📝" },
                { n: "2", title: "α-Step Delay", desc: `It takes α = ${alpha} periods for this contract to flow through procurement, inventory, and pricing systems.`, color: C.alpha, icon: "⏳" },
                { n: "3", title: "Price Adjusts", desc: `Sell price at time t+α finally reflects C(t). During the gap, sell may dip below C(t) — a margin violation.`, color: C.sell, icon: "💰" },
              ].map(step => (
                <div key={step.n} style={{ padding: 14, borderRadius: 10, background: step.color + "08", borderTop: `3px solid ${step.color}` }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 6 }}>
                    <span style={{ fontSize: 14 }}>{step.icon}</span>
                    <span style={{ fontSize: 12, fontWeight: 700, color: step.color }}>{step.title}</span>
                  </div>
                  <p style={{ fontSize: 11, color: C.muted, lineHeight: 1.6, margin: 0 }}>{step.desc}</p>
                </div>
              ))}
            </div>
            <div style={{
              marginTop: 14, padding: "10px 14px", borderRadius: 8,
              background: C.raised, border: `1px solid ${C.alpha}30`,
              fontFamily: font, fontSize: 11, color: C.muted, lineHeight: 1.6,
            }}>
              <strong style={{ color: C.alpha }}>Therefore:</strong> The effective constraint at forecast time t is not <code style={{ color: C.contract }}>contract(t)</code> but rather <code style={{ color: C.alpha }}>contract(t − α)</code>. Our model must learn this lag and enforce: <code style={{ color: C.constraint }}>sell_pred(t) ≥ contract(t − α) + margin</code>
            </div>
          </Card>
        </>)}

        {/* ═══════ ENCODER TAB ═══════ */}
        {tab === "encoder" && (<>
          <Card style={{ marginBottom: 14 }}>
            <Title icon="🔮" color={C.encoder}>Encoder: Reading the Past</Title>
            <p style={{ fontSize: 12, color: C.muted, margin: "0 0 12px", lineHeight: 1.7 }}>
              The encoder LSTM processes the <strong>historical window</strong> — both sell prices and contract values together. At each timestep, it receives a <Badge color={C.encoder}>2-dimensional input</Badge>: [sell(t), contract(t)].
            </p>

            <svg width={W} height={H - 20} viewBox={`0 0 ${W} ${H - 20}`} style={{ width: "100%", height: "auto" }}>
              {/* Encoder zone */}
              <rect x={pL} y={pT - 10} width={xS(SEQ_LEN - 1) - pL + 10} height={plotH - 20}
                fill={C.encoder} opacity={0.04} rx={6} stroke={C.encoder} strokeWidth={1} strokeDasharray="4 4" />
              <text x={pL + 8} y={pT + 4} fill={C.encoder} fontSize={10} fontFamily={font} fontWeight={600}>ENCODER WINDOW (T={SEQ_LEN})</text>

              {/* Grid */}
              {[0,1,2,3].map(i => {
                const v = minV + ((maxV - minV) * i) / 3;
                return (<g key={i}>
                  <line x1={pL} x2={xS(SEQ_LEN - 1) + 10} y1={yS(v)} y2={yS(v)} stroke={C.border} strokeWidth={0.5} />
                </g>);
              })}

              {/* Series */}
              <path d={contractData.slice(0, SEQ_LEN).map((v, i) => `${i === 0 ? "M" : "L"}${xS(i)},${yS(v)}`).join(" ")}
                fill="none" stroke={C.contract} strokeWidth={2} opacity={0.7} />
              <path d={sellData.map((v, i) => `${i === 0 ? "M" : "L"}${xS(i)},${yS(v)}`).join(" ")}
                fill="none" stroke={C.sell} strokeWidth={2} opacity={0.7} />

              {/* Active step highlight */}
              <line x1={xS(activeStep)} y1={pT - 10} x2={xS(activeStep)} y2={pT + plotH - 30}
                stroke={C.white} strokeWidth={1} opacity={0.2} />

              {/* Hidden state output arrow */}
              <defs>
                <marker id="arrE" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                  <path d="M0,0 L6,3 L0,6" fill={C.encoder} />
                </marker>
              </defs>
              <line x1={xS(SEQ_LEN - 1) + 15} y1={yS((maxV + minV) / 2)} x2={xS(SEQ_LEN - 1) + 60} y2={yS((maxV + minV) / 2)}
                stroke={C.encoder} strokeWidth={2.5} markerEnd="url(#arrE)" />
              <text x={xS(SEQ_LEN - 1) + 66} y={yS((maxV + minV) / 2) - 8} fill={C.encoder} fontSize={10} fontFamily={font} fontWeight={700}>
                h(T)
              </text>
              <text x={xS(SEQ_LEN - 1) + 66} y={yS((maxV + minV) / 2) + 12} fill={C.encoder} fontSize={10} fontFamily={font} fontWeight={700}>
                c(T)
              </text>
              <text x={xS(SEQ_LEN - 1) + 66} y={yS((maxV + minV) / 2) + 28} fill={C.dim} fontSize={8} fontFamily={font}>
                → Decoder
              </text>
            </svg>

            <input type="range" min={0} max={SEQ_LEN - 1} value={activeStep}
              onChange={e => { setActiveStep(Number(e.target.value)); setIsPlaying(false); }}
              style={{ width: "100%", accentColor: C.encoder, marginTop: 4 }} />
          </Card>

          <Card>
            <Title icon="🧠" color={C.encoder}>What the Encoder's Hidden State Captures</Title>
            <div style={{ fontSize: 12, color: C.muted, lineHeight: 1.8 }}>
              <p style={{ margin: "0 0 12px" }}>
                After processing all {SEQ_LEN} historical timesteps, the encoder's final <strong style={{ color: C.encoder }}>h(T)</strong> and <strong style={{ color: C.encoder }}>c(T)</strong> encode:
              </p>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                {[
                  { title: "Sell-Contract Spread", desc: "The typical margin between sell and contract, and how it changes over time.", color: C.sell, icon: "📏" },
                  { title: "α-Lag Pattern", desc: `The learned delay: sell prices adjust ~${alpha} steps after contract changes. The encoder sees this pattern repeatedly.`, color: C.alpha, icon: "⏱️" },
                  { title: "Trend & Level", desc: "Current price level, trajectory (rising/falling), and volatility of both series.", color: C.contract, icon: "📊" },
                  { title: "Violation History", desc: "Past instances where sell dipped below contract, helping the model understand risk zones.", color: C.red, icon: "⚠️" },
                ].map(item => (
                  <div key={item.title} style={{ padding: 12, borderRadius: 8, background: item.color + "08", border: `1px solid ${item.color}20` }}>
                    <div style={{ fontSize: 12, fontWeight: 700, color: item.color, marginBottom: 4 }}>{item.icon} {item.title}</div>
                    <div style={{ fontSize: 11, color: C.muted, lineHeight: 1.5 }}>{item.desc}</div>
                  </div>
                ))}
              </div>
              <p style={{ margin: "12px 0 0", padding: "10px 14px", borderRadius: 8, background: C.encoderDim, border: `1px solid ${C.encoder}25` }}>
                <strong style={{ color: C.encoder }}>Key:</strong> The encoder is a <em>Functional API</em> model using <code style={{ fontFamily: font, color: C.encoder }}>return_state=True</code> on the last LSTM layer. This gives us the explicit [h, c] state tensors to pass as <code style={{ fontFamily: font, color: C.decoder }}>initial_state</code> to the decoder.
              </p>
            </div>
          </Card>
        </>)}

        {/* ═══════ DECODER TAB ═══════ */}
        {tab === "decoder" && (<>
          <Card style={{ marginBottom: 14 }}>
            <Title icon="🎯" color={C.decoder}>Decoder: Generating the Future</Title>
            <p style={{ fontSize: 12, color: C.muted, margin: "0 0 14px", lineHeight: 1.7 }}>
              The decoder starts from the encoder's final state and steps forward through the forecast horizon. Its input is the <strong style={{ color: C.contract }}>known future contract values</strong> — contracts are signed ahead of time, so these are available at inference.
            </p>

            <svg width={W} height={H + 30} viewBox={`0 0 ${W} ${H + 30}`} style={{ width: "100%", height: "auto" }}>
              {/* Encoder zone (faded) */}
              <rect x={pL} y={pT} width={xS(SEQ_LEN - 1) - pL} height={plotH} fill={C.encoder} opacity={0.03} rx={4} />
              <text x={pL + 4} y={pT + 12} fill={C.encoder} fontSize={9} fontFamily={font} opacity={0.5}>ENCODER (history)</text>

              {/* Decoder zone */}
              <rect x={xS(SEQ_LEN - 0.5)} y={pT} width={xS(totalLen - 1) - xS(SEQ_LEN - 0.5) + 10} height={plotH}
                fill={C.decoder} opacity={0.06} rx={4} stroke={C.decoder} strokeWidth={1.5} strokeDasharray="4 4" />
              <text x={(xS(SEQ_LEN) + xS(totalLen - 1)) / 2} y={pT + 12} textAnchor="middle" fill={C.decoder} fontSize={10} fontFamily={font} fontWeight={600}>
                DECODER (H={FORECAST_H})
              </text>

              {/* Grid */}
              {[0,1,2,3,4].map(i => {
                const v = minV + ((maxV - minV) * i) / 4;
                return (<g key={i}>
                  <line x1={pL} x2={W - pR} y1={yS(v)} y2={yS(v)} stroke={C.border} strokeWidth={0.5} />
                  <text x={pL - 6} y={yS(v) + 3} textAnchor="end" fill={C.dim} fontSize={9} fontFamily={font}>${v.toFixed(0)}</text>
                </g>);
              })}

              {/* Historical series (faded) */}
              <path d={contractData.slice(0, SEQ_LEN).map((v, i) => `${i === 0 ? "M" : "L"}${xS(i)},${yS(v)}`).join(" ")}
                fill="none" stroke={C.contract} strokeWidth={1} opacity={0.25} />
              <path d={sellData.map((v, i) => `${i === 0 ? "M" : "L"}${xS(i)},${yS(v)}`).join(" ")}
                fill="none" stroke={C.sell} strokeWidth={1} opacity={0.25} />

              {/* Future contract (decoder input) */}
              <path d={[
                `M${xS(SEQ_LEN - 1)},${yS(contractData[SEQ_LEN - 1])}`,
                ...forecast.map((f, i) => `L${xS(SEQ_LEN + i)},${yS(f.contract)}`)
              ].join(" ")}
                fill="none" stroke={C.contract} strokeWidth={2.5} opacity={0.9}
                style={{ filter: `drop-shadow(0 0 4px ${C.contract}50)` }} />

              {/* Alpha-shifted floor */}
              <path d={forecast.map((f, i) => `${i === 0 ? "M" : "L"}${xS(SEQ_LEN + i)},${yS(f.floor)}`).join(" ")}
                fill="none" stroke={C.constraint} strokeWidth={2} strokeDasharray="6 3" opacity={0.7} />

              {/* Floor fill */}
              {forecast.map((f, i) => (
                <rect key={i} x={xS(SEQ_LEN + i) - plotW / totalLen / 2} y={yS(f.floor)}
                  width={plotW / totalLen} height={yS(minV) - yS(f.floor)}
                  fill={C.constraint} opacity={0.04} />
              ))}

              {/* Raw prediction */}
              <path d={[
                `M${xS(SEQ_LEN - 1)},${yS(sellData[SEQ_LEN - 1])}`,
                ...forecast.map((f, i) => `L${xS(SEQ_LEN + i)},${yS(f.rawPred)}`)
              ].join(" ")}
                fill="none" stroke={C.sell} strokeWidth={1.5} strokeDasharray="3 3" opacity={0.5} />

              {/* Constrained prediction */}
              <path d={[
                `M${xS(SEQ_LEN - 1)},${yS(sellData[SEQ_LEN - 1])}`,
                ...forecast.map((f, i) => `L${xS(SEQ_LEN + i)},${yS(f.constrained)}`)
              ].join(" ")}
                fill="none" stroke={C.decoder} strokeWidth={2.5}
                style={{ filter: `drop-shadow(0 0 6px ${C.decoder}50)` }} />

              {/* Forecast points */}
              {forecast.map((f, i) => (
                <g key={i}>
                  {f.wasViolation && (
                    <line x1={xS(SEQ_LEN + i)} y1={yS(f.rawPred)} x2={xS(SEQ_LEN + i)} y2={yS(f.constrained)}
                      stroke={C.constraint} strokeWidth={2} opacity={0.6} />
                  )}
                  <circle cx={xS(SEQ_LEN + i)} cy={yS(f.constrained)} r={4} fill={C.decoder}
                    style={{ filter: `drop-shadow(0 0 6px ${C.decoder})` }} />
                  {f.wasViolation && (
                    <circle cx={xS(SEQ_LEN + i)} cy={yS(f.rawPred)} r={3} fill={C.red} opacity={0.7} />
                  )}
                  <circle cx={xS(SEQ_LEN + i)} cy={yS(f.contract)} r={3} fill={C.contract} />
                </g>
              ))}

              {/* State transfer arrow */}
              <defs>
                <marker id="arrD" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
                  <path d="M0,0 L7,3.5 L0,7" fill={C.encoder} />
                </marker>
              </defs>
              <path d={`M${xS(SEQ_LEN - 1)},${H + 6} C${xS(SEQ_LEN - 1) + 30},${H + 6} ${xS(SEQ_LEN) - 30},${H + 6} ${xS(SEQ_LEN)},${H + 6}`}
                fill="none" stroke={C.encoder} strokeWidth={2} markerEnd="url(#arrD)" />
              <text x={(xS(SEQ_LEN - 1) + xS(SEQ_LEN)) / 2} y={H + 22} textAnchor="middle" fill={C.encoder} fontSize={9} fontFamily={font} fontWeight={600}>
                [h(T), c(T)] → initial_state
              </text>

              {/* X labels */}
              {Array.from({ length: totalLen }, (_, i) => (
                <text key={i} x={xS(i)} y={H - 2} textAnchor="middle"
                  fill={i >= SEQ_LEN ? C.decoder : C.dim} fontSize={7} fontFamily={font}>
                  {i >= SEQ_LEN ? `F${i - SEQ_LEN + 1}` : `M${i + 1}`}
                </text>
              ))}

              {/* Legend */}
              <g transform="translate(60, 28)">
                <line x1={0} x2={16} y1={0} y2={0} stroke={C.contract} strokeWidth={2.5} />
                <text x={20} y={3} fill={C.contract} fontSize={8} fontFamily={font}>Future Contract (known)</text>
                <line x1={170} x2={186} y1={0} y2={0} stroke={C.constraint} strokeWidth={2} strokeDasharray="6 3" />
                <text x={190} y={3} fill={C.constraint} fontSize={8} fontFamily={font}>Floor (α-shifted)</text>
                <line x1={310} x2={326} y1={0} y2={0} stroke={C.sell} strokeWidth={1.5} strokeDasharray="3 3" />
                <text x={330} y={3} fill={C.sell} fontSize={8} fontFamily={font}>Raw pred</text>
                <line x1={400} x2={416} y1={0} y2={0} stroke={C.decoder} strokeWidth={2.5} />
                <text x={420} y={3} fill={C.decoder} fontSize={8} fontFamily={font}>Constrained pred</text>
              </g>
            </svg>
          </Card>

          <Card>
            <Title icon="🔑" color={C.decoder}>Why Encoder-Decoder?</Title>
            <p style={{ fontSize: 12, color: C.muted, lineHeight: 1.7, margin: "0 0 12px" }}>
              A standard single LSTM can't easily incorporate <em>known future inputs</em>. The encoder-decoder architecture solves this elegantly:
            </p>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
              <div style={{ padding: 14, borderRadius: 10, background: C.encoderDim, borderLeft: `3px solid ${C.encoder}` }}>
                <div style={{ fontSize: 12, fontWeight: 700, color: C.encoder, marginBottom: 4 }}>Encoder</div>
                <div style={{ fontSize: 11, color: C.muted, lineHeight: 1.6 }}>
                  Reads [sell, contract] history. Compresses all temporal dynamics — the α-lag, spread patterns, violation history — into h(T), c(T).
                </div>
              </div>
              <div style={{ padding: 14, borderRadius: 10, background: C.decoderDim, borderLeft: `3px solid ${C.decoder}` }}>
                <div style={{ fontSize: 12, fontWeight: 700, color: C.decoder, marginBottom: 4 }}>Decoder</div>
                <div style={{ fontSize: 11, color: C.muted, lineHeight: 1.6 }}>
                  Starts from encoder state. At each future step, it receives the <strong>known contract value</strong> and generates a sell price prediction — conditioned on both past context and future contracts.
                </div>
              </div>
            </div>
          </Card>
        </>)}

        {/* ═══════ CONSTRAINT TAB ═══════ */}
        {tab === "constraint" && (<>
          <Card style={{ marginBottom: 14 }}>
            <Title icon="🔒" color={C.constraint}>The Constraint Mechanism</Title>
            <p style={{ fontSize: 12, color: C.muted, margin: "0 0 16px", lineHeight: 1.7 }}>
              The constraint is enforced in <strong>two complementary ways</strong>: a hard architectural constraint at the output layer, and a soft penalty in the loss function. This dual approach ensures the constraint holds even during early training.
            </p>

            {/* Constraint flow */}
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              {[
                {
                  n: "1", title: "Decoder produces raw prediction",
                  formula: "raw_pred(t) = Dense(h_dec(t))",
                  desc: "The decoder's hidden state is transformed via a linear layer into an unconstrained sell price prediction.",
                  color: C.sell,
                },
                {
                  n: "2", title: "Compute α-shifted floor",
                  formula: `floor(t) = contract(t − α) + margin   [α = ${alpha}]`,
                  desc: `The effective contract floor accounts for the ${alpha}-step propagation delay. "margin" is a minimum buffer (e.g., $2) to maintain profitability.`,
                  color: C.alpha,
                },
                {
                  n: "3", title: "Apply soft-max constraint (training)",
                  formula: "pred(t) = LogSumExp(raw_pred(t), floor(t)) / τ",
                  desc: "LogSumExp is a smooth, differentiable approximation of max(). Temperature τ controls sharpness — as τ→∞ it approaches hard max. This lets gradients flow through the constraint during backpropagation.",
                  color: C.constraint,
                },
                {
                  n: "4", title: "Apply hard max (inference)",
                  formula: "pred(t) = max(raw_pred(t), floor(t))",
                  desc: "At inference time, we switch to a hard max for exact constraint enforcement. The model was trained with the soft version, so it already learned to stay above the floor — the hard max is a safety net.",
                  color: C.decoder,
                },
              ].map(step => (
                <div key={step.n} style={{
                  display: "flex", gap: 14, padding: 14, borderRadius: 10,
                  background: step.color + "06", borderLeft: `3px solid ${step.color}`,
                }}>
                  <div style={{
                    width: 28, height: 28, borderRadius: "50%", background: step.color,
                    color: "#000", display: "flex", alignItems: "center", justifyContent: "center",
                    fontSize: 14, fontWeight: 700, fontFamily: font, flexShrink: 0,
                  }}>{step.n}</div>
                  <div>
                    <div style={{ fontSize: 13, fontWeight: 700, color: step.color, marginBottom: 2 }}>{step.title}</div>
                    <code style={{
                      display: "block", padding: "4px 8px", borderRadius: 4,
                      background: C.raised, fontSize: 11, fontFamily: font,
                      color: step.color, marginBottom: 4, width: "fit-content",
                    }}>{step.formula}</code>
                    <div style={{ fontSize: 11, color: C.muted, lineHeight: 1.6 }}>{step.desc}</div>
                  </div>
                </div>
              ))}
            </div>
          </Card>

          {/* Why not just add penalty to loss? */}
          <Card>
            <Title icon="⚖️" color={C.amber}>Why Not Just Use a Loss Penalty?</Title>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
              <div style={{ padding: 14, borderRadius: 10, background: C.redDim, borderTop: `3px solid ${C.red}` }}>
                <div style={{ fontSize: 12, fontWeight: 700, color: C.red, marginBottom: 6 }}>❌ Loss penalty alone</div>
                <div style={{ fontSize: 11, color: C.muted, lineHeight: 1.6 }}>
                  A penalty term like <code style={{ fontFamily: font, color: C.red }}>λ · max(0, floor − pred)²</code> discourages violations but <strong>does not guarantee</strong> them. At inference, the model can still predict below the floor. Tuning λ is fragile.
                </div>
              </div>
              <div style={{ padding: 14, borderRadius: 10, background: C.decoderDim, borderTop: `3px solid ${C.decoder}` }}>
                <div style={{ fontSize: 12, fontWeight: 700, color: C.decoder, marginBottom: 6 }}>✓ Architectural constraint + penalty</div>
                <div style={{ fontSize: 11, color: C.muted, lineHeight: 1.6 }}>
                  The <code style={{ fontFamily: font, color: C.constraint }}>max()</code> in the output layer <strong>physically prevents</strong> violations — it's mathematically impossible for the output to be below the floor. The loss penalty provides gradient signal during training to help the raw prediction learn to stay above naturally.
                </div>
              </div>
            </div>
          </Card>
        </>)}

        {/* ═══════ ARCHITECTURE TAB ═══════ */}
        {tab === "arch" && (<>
          <Card style={{ marginBottom: 14 }}>
            <Title icon="⚙️" color={C.contract}>Keras Model — Layer by Layer</Title>
            <p style={{ fontSize: 12, color: C.muted, margin: "0 0 14px", lineHeight: 1.6 }}>
              Interactive architecture breakdown. <Badge color={C.encoder}>Hover or click</Badge> any layer for details, shapes, and parameter derivations.
            </p>

            {/* Column headers */}
            <div style={{
              display: "grid", gridTemplateColumns: "155px 1fr 100px 70px",
              gap: 10, padding: "4px 12px", marginBottom: 6,
            }}>
              <span style={{ fontSize: 9, color: C.dim, fontFamily: font, letterSpacing: "0.5px" }}>LAYER</span>
              <span style={{ fontSize: 9, color: C.dim, fontFamily: font, letterSpacing: "0.5px" }}>KERAS CODE</span>
              <span style={{ fontSize: 9, color: C.dim, fontFamily: font, letterSpacing: "0.5px" }}>OUTPUT SHAPE</span>
              <span style={{ fontSize: 9, color: C.dim, fontFamily: font, letterSpacing: "0.5px", textAlign: "right" }}>PARAMS</span>
            </div>

            <KerasArchPane alpha={alpha} />
          </Card>

          {/* Data flow */}
          <Card>
            <Title icon="🔄" color={C.encoder}>Encoder-Decoder Data Flow</Title>
            <svg width={720} height={180} viewBox="0 0 720 180" style={{ width: "100%", height: "auto" }}>
              {/* Encoder box */}
              <rect x={20} y={30} width={240} height={100} rx={10} fill={C.encoder + "0a"} stroke={C.encoder} strokeWidth={1.5} />
              <text x={140} y={22} textAnchor="middle" fill={C.encoder} fontSize={11} fontFamily={font} fontWeight={700}>ENCODER</text>

              {/* Encoder input */}
              <rect x={40} y={50} width={90} height={30} rx={6} fill={C.contract + "15"} stroke={C.contract} strokeWidth={1} />
              <text x={85} y={69} textAnchor="middle" fill={C.contract} fontSize={9} fontFamily={font}>[sell, contract]</text>
              <text x={85} y={97} textAnchor="middle" fill={C.dim} fontSize={8} fontFamily={font}>(T, 2)</text>

              {/* LSTM blocks */}
              <rect x={150} y={50} width={45} height={30} rx={4} fill={C.encoder + "20"} stroke={C.encoder} strokeWidth={1} />
              <text x={172} y={69} textAnchor="middle" fill={C.encoder} fontSize={8} fontFamily={font}>LSTM</text>
              <rect x={200} y={50} width={45} height={30} rx={4} fill={C.encoder + "20"} stroke={C.encoder} strokeWidth={1} />
              <text x={222} y={69} textAnchor="middle" fill={C.encoder} fontSize={8} fontFamily={font}>LSTM</text>

              {/* Arrow encoder → decoder */}
              <defs>
                <marker id="arrFlow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
                  <path d="M0,0 L8,4 L0,8" fill={C.amber} />
                </marker>
              </defs>
              <line x1={260} y1={80} x2={310} y2={80} stroke={C.amber} strokeWidth={2.5} markerEnd="url(#arrFlow)" />
              <text x={285} y={72} textAnchor="middle" fill={C.amber} fontSize={8} fontFamily={font} fontWeight={700}>h,c</text>

              {/* Decoder box */}
              <rect x={320} y={30} width={200} height={100} rx={10} fill={C.decoder + "0a"} stroke={C.decoder} strokeWidth={1.5} />
              <text x={420} y={22} textAnchor="middle" fill={C.decoder} fontSize={11} fontFamily={font} fontWeight={700}>DECODER</text>

              {/* Decoder input */}
              <rect x={335} y={50} width={70} height={30} rx={6} fill={C.contract + "15"} stroke={C.contract} strokeWidth={1} />
              <text x={370} y={65} textAnchor="middle" fill={C.contract} fontSize={8} fontFamily={font}>future</text>
              <text x={370} y={75} textAnchor="middle" fill={C.contract} fontSize={8} fontFamily={font}>contract</text>
              <text x={370} y={97} textAnchor="middle" fill={C.dim} fontSize={8} fontFamily={font}>(H, 1)</text>

              {/* LSTM + Dense */}
              <rect x={420} y={50} width={40} height={30} rx={4} fill={C.decoder + "20"} stroke={C.decoder} strokeWidth={1} />
              <text x={440} y={69} textAnchor="middle" fill={C.decoder} fontSize={8} fontFamily={font}>LSTM</text>
              <rect x={465} y={50} width={40} height={30} rx={4} fill={C.sell + "20"} stroke={C.sell} strokeWidth={1} />
              <text x={485} y={69} textAnchor="middle" fill={C.sell} fontSize={8} fontFamily={font}>Dense</text>

              {/* Arrow decoder → constraint */}
              <line x1={520} y1={80} x2={560} y2={80} stroke={C.constraint} strokeWidth={2} markerEnd="url(#arrFlow)" />

              {/* Constraint box */}
              <rect x={570} y={30} width={130} height={100} rx={10} fill={C.constraint + "0a"} stroke={C.constraint} strokeWidth={1.5} />
              <text x={635} y={22} textAnchor="middle" fill={C.constraint} fontSize={11} fontFamily={font} fontWeight={700}>CONSTRAINT</text>
              <text x={635} y={60} textAnchor="middle" fill={C.alpha} fontSize={9} fontFamily={font}>α-shift</text>
              <text x={635} y={78} textAnchor="middle" fill={C.constraint} fontSize={10} fontFamily={font} fontWeight={700}>max(pred, floor)</text>
              <text x={635} y={95} textAnchor="middle" fill={C.dim} fontSize={8} fontFamily={font}>(H, 1)</text>

              {/* Known contract into constraint */}
              <path d="M370,100 L370,155 L600,155 L600,130" fill="none" stroke={C.contract} strokeWidth={1} strokeDasharray="3 3" opacity={0.5} />
              <text x={485} y={148} textAnchor="middle" fill={C.contract} fontSize={8} fontFamily={font} opacity={0.6}>known contract → α-shift → floor</text>
            </svg>
          </Card>
        </>)}

        {/* ═══════ FORECAST TAB ═══════ */}
        {tab === "forecast" && (<>
          <Card style={{ marginBottom: 14 }}>
            <Title icon="📈" color={C.decoder}>Constrained Forecast Results</Title>

            <svg width={W} height={H + 10} viewBox={`0 0 ${W} ${H + 10}`} style={{ width: "100%", height: "auto" }}>
              {/* Grid */}
              {[0,1,2,3,4].map(i => {
                const v = minV + ((maxV - minV) * i) / 4;
                return (<g key={i}>
                  <line x1={pL} x2={W - pR} y1={yS(v)} y2={yS(v)} stroke={C.border} strokeWidth={0.5} />
                  <text x={pL - 6} y={yS(v) + 3} textAnchor="end" fill={C.dim} fontSize={9} fontFamily={font}>${v.toFixed(0)}</text>
                </g>);
              })}

              {/* Forecast zone */}
              <rect x={xS(SEQ_LEN - 0.5)} y={pT} width={xS(totalLen - 1) - xS(SEQ_LEN - 0.5) + 10} height={plotH}
                fill={C.decoder} opacity={0.04} rx={4} stroke={C.decoder} strokeWidth={1} strokeDasharray="4 4" />

              {/* Historical lines */}
              <path d={contractData.slice(0, SEQ_LEN).map((v, i) => `${i === 0 ? "M" : "L"}${xS(i)},${yS(v)}`).join(" ")}
                fill="none" stroke={C.contract} strokeWidth={1.5} opacity={0.6} />
              <path d={sellData.map((v, i) => `${i === 0 ? "M" : "L"}${xS(i)},${yS(v)}`).join(" ")}
                fill="none" stroke={C.sell} strokeWidth={1.5} opacity={0.6} />

              {/* Future contract */}
              <path d={[`M${xS(SEQ_LEN - 1)},${yS(contractData[SEQ_LEN - 1])}`,
                ...forecast.map((f, i) => `L${xS(SEQ_LEN + i)},${yS(f.contract)}`)].join(" ")}
                fill="none" stroke={C.contract} strokeWidth={2.5} />

              {/* Floor */}
              <path d={forecast.map((f, i) => `${i === 0 ? "M" : "L"}${xS(SEQ_LEN + i)},${yS(f.floor)}`).join(" ")}
                fill="none" stroke={C.constraint} strokeWidth={2} strokeDasharray="6 3" />

              {/* Floor fill */}
              <path d={[
                ...forecast.map((f, i) => `${i === 0 ? "M" : "L"}${xS(SEQ_LEN + i)},${yS(f.floor)}`),
                `L${xS(totalLen - 1)},${yS(minV)}`,
                `L${xS(SEQ_LEN)},${yS(minV)}`, "Z"
              ].join(" ")} fill={C.constraint} opacity={0.05} />

              {/* Raw prediction (where violations happen) */}
              <path d={[`M${xS(SEQ_LEN - 1)},${yS(sellData[SEQ_LEN - 1])}`,
                ...forecast.map((f, i) => `L${xS(SEQ_LEN + i)},${yS(f.rawPred)}`)].join(" ")}
                fill="none" stroke={C.red} strokeWidth={1.5} strokeDasharray="3 3" opacity={0.5} />

              {/* Constrained prediction */}
              <path d={[`M${xS(SEQ_LEN - 1)},${yS(sellData[SEQ_LEN - 1])}`,
                ...forecast.map((f, i) => `L${xS(SEQ_LEN + i)},${yS(f.constrained)}`)].join(" ")}
                fill="none" stroke={C.decoder} strokeWidth={3}
                style={{ filter: `drop-shadow(0 0 8px ${C.decoder}60)` }} />

              {/* Points and corrections */}
              {forecast.map((f, i) => (
                <g key={i}>
                  {f.wasViolation && (
                    <>
                      <line x1={xS(SEQ_LEN + i)} y1={yS(f.rawPred)} x2={xS(SEQ_LEN + i)} y2={yS(f.constrained)}
                        stroke={C.constraint} strokeWidth={2.5} opacity={0.5} />
                      <circle cx={xS(SEQ_LEN + i)} cy={yS(f.rawPred)} r={4} fill={C.red} opacity={0.7}
                        style={{ filter: `drop-shadow(0 0 4px ${C.red})` }} />
                      <text x={xS(SEQ_LEN + i) - 12} y={(yS(f.rawPred) + yS(f.constrained)) / 2 + 3}
                        fill={C.constraint} fontSize={8} fontFamily={font} fontWeight={600}>↑</text>
                    </>
                  )}
                  <circle cx={xS(SEQ_LEN + i)} cy={yS(f.constrained)} r={5} fill={C.decoder}
                    style={{ filter: `drop-shadow(0 0 8px ${C.decoder})` }} />
                  <text x={xS(SEQ_LEN + i)} y={yS(f.constrained) - 10} textAnchor="middle"
                    fill={C.decoder} fontSize={9} fontFamily={font} fontWeight={600}>${f.constrained}</text>
                </g>
              ))}

              {/* X labels */}
              {Array.from({ length: totalLen }, (_, i) => (
                <text key={i} x={xS(i)} y={H + 4} textAnchor="middle"
                  fill={i >= SEQ_LEN ? C.decoder : C.dim} fontSize={7} fontFamily={font}>
                  {i >= SEQ_LEN ? `F${i - SEQ_LEN + 1}` : `M${i + 1}`}
                </text>
              ))}

              {/* Legend */}
              <g transform={`translate(60, ${pT + 6})`}>
                <line x1={0} x2={16} y1={0} y2={0} stroke={C.contract} strokeWidth={2.5} />
                <text x={20} y={3} fill={C.contract} fontSize={8} fontFamily={font}>Contract</text>
                <line x1={90} x2={106} y1={0} y2={0} stroke={C.constraint} strokeWidth={2} strokeDasharray="5 3" />
                <text x={110} y={3} fill={C.constraint} fontSize={8} fontFamily={font}>Floor (α-shifted)</text>
                <line x1={220} x2={236} y1={0} y2={0} stroke={C.red} strokeWidth={1.5} strokeDasharray="3 3" />
                <text x={240} y={3} fill={C.red} fontSize={8} fontFamily={font}>Raw (unconstrained)</text>
                <line x1={360} x2={376} y1={0} y2={0} stroke={C.decoder} strokeWidth={3} />
                <text x={380} y={3} fill={C.decoder} fontSize={8} fontFamily={font}>Final (constrained)</text>
              </g>
            </svg>
          </Card>

          {/* Forecast table */}
          <Card style={{ marginBottom: 14 }}>
            <Title icon="📋" color={C.decoder}>Forecast Detail</Title>
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11, fontFamily: font }}>
                <thead>
                  <tr>
                    {["Step", "Contract", `Floor (t−${alpha})`, "Raw Pred", "Constrained", "Status"].map(h => (
                      <th key={h} style={{ padding: "8px 10px", textAlign: "center", color: C.dim, borderBottom: `1px solid ${C.border}`, fontSize: 10 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {forecast.map((f, i) => (
                    <tr key={i} style={{ background: f.wasViolation ? C.redDim : "transparent" }}>
                      <td style={{ padding: "6px 10px", textAlign: "center", color: C.decoder, fontWeight: 600 }}>F{i + 1}</td>
                      <td style={{ padding: "6px 10px", textAlign: "center", color: C.contract }}>${f.contract}</td>
                      <td style={{ padding: "6px 10px", textAlign: "center", color: C.constraint }}>${f.floor.toFixed(2)}</td>
                      <td style={{ padding: "6px 10px", textAlign: "center", color: f.wasViolation ? C.red : C.muted }}>
                        ${f.rawPred}{f.wasViolation && " ⚠️"}
                      </td>
                      <td style={{ padding: "6px 10px", textAlign: "center", color: C.decoder, fontWeight: 700 }}>${f.constrained}</td>
                      <td style={{ padding: "6px 10px", textAlign: "center" }}>
                        <span style={{
                          padding: "2px 8px", borderRadius: 10, fontSize: 9, fontWeight: 700,
                          background: f.wasViolation ? C.constraint + "20" : C.decoder + "20",
                          color: f.wasViolation ? C.constraint : C.decoder,
                        }}>
                          {f.wasViolation ? "CORRECTED ↑" : "OK ✓"}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Summary */}
          <Card glow={C.decoder + "10"}>
            <Title icon="🎯" color={C.amber}>End-to-End Summary</Title>
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              {[
                { label: "Encoder reads history", detail: `[sell, contract] over T=${SEQ_LEN} steps → learns the α=${alpha} lag, spread dynamics, and violation patterns → outputs [h(T), c(T)]`, color: C.encoder },
                { label: "Decoder generates future", detail: `Initialized with [h(T), c(T)], fed known future contract values over H=${FORECAST_H} steps → produces raw sell predictions`, color: C.decoder },
                { label: "α-shift computes effective floor", detail: `floor(t) = contract(t − ${alpha}) + margin. Accounts for the ${alpha}-step delay before contract changes affect sell pricing`, color: C.alpha },
                { label: "Constraint enforces floor", detail: "final_pred(t) = max(raw_pred(t), floor(t)). Soft (LogSumExp) during training for differentiability, hard (max) at inference for guaranteed compliance", color: C.constraint },
              ].map((step, i) => (
                <div key={i} style={{
                  display: "flex", gap: 12, padding: "10px 14px", borderRadius: 8,
                  background: step.color + "06", borderLeft: `3px solid ${step.color}`,
                }}>
                  <div style={{
                    width: 24, height: 24, borderRadius: "50%", background: step.color,
                    color: "#000", display: "flex", alignItems: "center", justifyContent: "center",
                    fontSize: 12, fontWeight: 700, fontFamily: font, flexShrink: 0,
                  }}>{i + 1}</div>
                  <div>
                    <span style={{ fontSize: 12, fontWeight: 700, color: step.color }}>{step.label}</span>
                    <p style={{ fontSize: 11, color: C.muted, margin: "2px 0 0", lineHeight: 1.5 }}>{step.detail}</p>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </>)}
      </div>
    </div>
  );
}
