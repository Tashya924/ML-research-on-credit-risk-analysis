"use client";

import { useState, useRef } from "react";

/* ─────────────────────────────────────────────────────────────
   Feature Configuration
   ───────────────────────────────────────────────────────────── */

const FEATURE_GROUPS = [
  {
    label: "Personal Information",
    fields: [
      {
        name: "LIMIT_BAL",
        label: "Credit Limit",
        type: "number",
        placeholder: "e.g. 200000",
        hint: "Amount of credit in NT dollars",
        defaultValue: 200000,
      },
      {
        name: "SEX",
        label: "Gender",
        type: "select",
        options: [
          { value: 1, label: "Male" },
          { value: 2, label: "Female" },
        ],
        defaultValue: 1,
      },
      {
        name: "EDUCATION",
        label: "Education Level",
        type: "select",
        options: [
          { value: 1, label: "Graduate School" },
          { value: 2, label: "University" },
          { value: 3, label: "High School" },
          { value: 4, label: "Others" },
        ],
        defaultValue: 2,
      },
      {
        name: "MARRIAGE",
        label: "Marital Status",
        type: "select",
        options: [
          { value: 1, label: "Married" },
          { value: 2, label: "Single" },
          { value: 3, label: "Others" },
        ],
        defaultValue: 1,
      },
      {
        name: "AGE",
        label: "Age",
        type: "number",
        placeholder: "e.g. 35",
        hint: "Age in years",
        defaultValue: 35,
      },
    ],
    columns: "form-grid-5",
  },
  {
    label: "Repayment History",
    sublabel: "(-2 = No consumption, -1 = Paid in full, 0 = Revolving credit, 1–8 = Months delayed)",
    fields: [
      { name: "PAY_0", label: "September", type: "number", placeholder: "0", hint: "PAY_0", defaultValue: 0 },
      { name: "PAY_2", label: "August", type: "number", placeholder: "0", hint: "PAY_2", defaultValue: 0 },
      { name: "PAY_3", label: "July", type: "number", placeholder: "0", hint: "PAY_3", defaultValue: 0 },
      { name: "PAY_4", label: "June", type: "number", placeholder: "0", hint: "PAY_4", defaultValue: 0 },
      { name: "PAY_5", label: "May", type: "number", placeholder: "0", hint: "PAY_5", defaultValue: 0 },
      { name: "PAY_6", label: "April", type: "number", placeholder: "0", hint: "PAY_6", defaultValue: 0 },
    ],
    columns: "form-grid-6",
  },
  {
    label: "Bill Amounts",
    sublabel: "Amount of bill statement in NT dollars",
    fields: [
      { name: "BILL_AMT1", label: "September", type: "number", placeholder: "50000", hint: "BILL_AMT1", defaultValue: 50000 },
      { name: "BILL_AMT2", label: "August", type: "number", placeholder: "48000", hint: "BILL_AMT2", defaultValue: 48000 },
      { name: "BILL_AMT3", label: "July", type: "number", placeholder: "45000", hint: "BILL_AMT3", defaultValue: 45000 },
      { name: "BILL_AMT4", label: "June", type: "number", placeholder: "42000", hint: "BILL_AMT4", defaultValue: 42000 },
      { name: "BILL_AMT5", label: "May", type: "number", placeholder: "40000", hint: "BILL_AMT5", defaultValue: 40000 },
      { name: "BILL_AMT6", label: "April", type: "number", placeholder: "38000", hint: "BILL_AMT6", defaultValue: 38000 },
    ],
    columns: "form-grid-6",
  },
  {
    label: "Payment Amounts",
    sublabel: "Amount of previous payment in NT dollars",
    fields: [
      { name: "PAY_AMT1", label: "September", type: "number", placeholder: "5000", hint: "PAY_AMT1", defaultValue: 5000 },
      { name: "PAY_AMT2", label: "August", type: "number", placeholder: "5000", hint: "PAY_AMT2", defaultValue: 5000 },
      { name: "PAY_AMT3", label: "July", type: "number", placeholder: "5000", hint: "PAY_AMT3", defaultValue: 5000 },
      { name: "PAY_AMT4", label: "June", type: "number", placeholder: "5000", hint: "PAY_AMT4", defaultValue: 5000 },
      { name: "PAY_AMT5", label: "May", type: "number", placeholder: "5000", hint: "PAY_AMT5", defaultValue: 5000 },
      { name: "PAY_AMT6", label: "April", type: "number", placeholder: "5000", hint: "PAY_AMT6", defaultValue: 5000 },
    ],
    columns: "form-grid-6",
  },
];

/* ─────────────────────────────────────────────────────────────
   Helper Functions
   ───────────────────────────────────────────────────────────── */

function getDefaultValues() {
  const defaults = {};
  FEATURE_GROUPS.forEach((group) => {
    group.fields.forEach((field) => {
      defaults[field.name] = field.defaultValue;
    });
  });
  return defaults;
}

function getRiskColor(flag) {
  switch (flag) {
    case "LOW": return "#228B22";
    case "MODERATE": return "#DAA520";
    case "HIGH": return "#B22222";
    default: return "#6B6B6B";
  }
}

function getRiskEmoji(flag) {
  switch (flag) {
    case "LOW": return "●";
    case "MODERATE": return "●";
    case "HIGH": return "●";
    default: return "●";
  }
}

/* ─────────────────────────────────────────────────────────────
   Section Label Component
   ───────────────────────────────────────────────────────────── */
function SectionLabel({ text }) {
  return (
    <div className="section-label">
      <span>{text}</span>
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────
   Main Page Component
   ───────────────────────────────────────────────────────────── */
export default function Home() {
  const [formData, setFormData] = useState(getDefaultValues());
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const resultsRef = useRef(null);

  const handleChange = (name, value) => {
    setFormData((prev) => ({ ...prev, [name]: Number(value) }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Prediction failed");
      }

      setResults(data);

      // Scroll to results after a brief delay for render
      setTimeout(() => {
        resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 100);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFormData(getDefaultValues());
    setResults(null);
    setError(null);
  };

  return (
    <main>
      {/* ── Hero ── */}
      <section className="hero">
        <div className="container-narrow">
          <div className="small-caps" style={{ marginBottom: "1.5rem" }}>
            Machine Learning Ensemble
          </div>
          <h1>Credit Risk Analysis</h1>
          <div className="hero-rule" />
          <p>
            Evaluate credit default probability through fifteen distinct machine learning models.
            Input customer financial data and receive comprehensive risk assessments with
            feature-level importance analysis — identifying the precise parameters driving risk.
          </p>
        </div>
      </section>

      <hr className="rule" />

      {/* ── Input Form ── */}
      <section style={{ padding: "4rem 0" }}>
        <div className="container-narrow">
          <SectionLabel text="Customer Data Input" />

          <form onSubmit={handleSubmit}>
            {FEATURE_GROUPS.map((group) => (
              <div className="form-section" key={group.label}>
                <h3 style={{ marginBottom: "0.25rem" }}>{group.label}</h3>
                {group.sublabel && (
                  <p style={{ fontSize: "0.8125rem", marginBottom: "1.25rem", opacity: 0.7 }}>
                    {group.sublabel}
                  </p>
                )}

                <div className={`form-grid ${group.columns}`}>
                  {group.fields.map((field) => (
                    <div className="input-group" key={field.name}>
                      <label className="input-label" htmlFor={field.name}>
                        {field.label}
                      </label>
                      {field.type === "select" ? (
                        <select
                          id={field.name}
                          className="input"
                          value={formData[field.name]}
                          onChange={(e) => handleChange(field.name, e.target.value)}
                        >
                          {field.options.map((opt) => (
                            <option key={opt.value} value={opt.value}>
                              {opt.label}
                            </option>
                          ))}
                        </select>
                      ) : (
                        <input
                          id={field.name}
                          type="number"
                          className="input"
                          placeholder={field.placeholder}
                          value={formData[field.name]}
                          onChange={(e) => handleChange(field.name, e.target.value)}
                          step="any"
                        />
                      )}
                      {field.hint && (
                        <span className="input-hint">{field.hint}</span>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            ))}

            {/* ── Actions ── */}
            <div style={{ display: "flex", gap: "1rem", marginTop: "1rem" }}>
              <button type="submit" className="btn btn-primary" disabled={loading}>
                {loading ? (
                  <>
                    <span className="spinner" />
                    Analyzing…
                  </>
                ) : (
                  "Run Risk Analysis"
                )}
              </button>
              <button type="button" className="btn btn-secondary" onClick={handleReset}>
                Reset to Defaults
              </button>
            </div>

            {error && (
              <div
                style={{
                  marginTop: "1.5rem",
                  padding: "1rem 1.5rem",
                  background: "rgba(178, 34, 34, 0.06)",
                  border: "1px solid rgba(178, 34, 34, 0.2)",
                  borderRadius: "var(--radius-md)",
                  color: "#a02020",
                  fontSize: "0.9375rem",
                }}
              >
                {error}
              </div>
            )}
          </form>
        </div>
      </section>

      {/* ── Results ── */}
      {results && (
        <div ref={resultsRef}>
          <hr className="rule" />

          {/* Consensus Summary */}
          <section style={{ padding: "4rem 0 3rem" }} className="fade-in">
            <div className="container-narrow">
              <SectionLabel text="Risk Assessment Summary" />

              <div className={`card card-elevated consensus-card consensus-${results.summary.consensus_risk.toLowerCase()}`}>
                <div className="small-caps" style={{ marginBottom: "1rem", color: "var(--muted-foreground)" }}>
                  Ensemble Consensus
                </div>
                <div className="consensus-number">
                  {(results.summary.average_default_probability * 100).toFixed(1)}%
                </div>
                <p style={{
                  fontFamily: "var(--font-display)",
                  fontSize: "1.25rem",
                  color: "var(--foreground)",
                  margin: "0.75rem auto 0",
                }}>
                  Average Default Probability
                </p>
                <div style={{ marginTop: "1.5rem" }}>
                  <span className={`badge badge-${results.summary.consensus_risk.toLowerCase()}`}>
                    <span style={{ color: getRiskColor(results.summary.consensus_risk) }}>
                      {getRiskEmoji(results.summary.consensus_risk)}
                    </span>
                    {results.summary.consensus_risk} Risk
                  </span>
                </div>
              </div>

              {/* Stats Bar */}
              <div className="stats-bar mt-8 fade-in stagger-1">
                <div className="stat-item">
                  <div className="stat-number">{results.summary.total_models}</div>
                  <div className="stat-label">Models Evaluated</div>
                </div>
                <div className="stat-item">
                  <div className="stat-number" style={{ color: "#B22222" }}>
                    {results.summary.models_predicting_default}
                  </div>
                  <div className="stat-label">Predict Default</div>
                </div>
                <div className="stat-item">
                  <div className="stat-number" style={{ color: "#228B22" }}>
                    {results.summary.models_predicting_no_default}
                  </div>
                  <div className="stat-label">Predict No Default</div>
                </div>
                <div className="stat-item">
                  <div className="stat-number" style={{ color: "var(--accent)" }}>
                    {(results.summary.average_default_probability * 100).toFixed(1)}
                    <span style={{ fontSize: "1.5rem" }}>%</span>
                  </div>
                  <div className="stat-label">Avg. Probability</div>
                </div>
              </div>
            </div>
          </section>

          {/* ── Worst Parameters ── */}
          <section style={{ padding: "3rem 0" }} className="fade-in stagger-2">
            <div className="container-narrow">
              <SectionLabel text="Worst Risk Parameters" />

              <p style={{ marginBottom: "2rem", fontSize: "1.0625rem" }}>
                Features with the highest average importance across all models — 
                the primary drivers of the risk assessment. These parameters have 
                the greatest influence on whether a customer is predicted to default.
              </p>

              <div className="card card-accent card-elevated" style={{ padding: "0" }}>
                <div style={{ padding: "1.5rem 2rem 0.75rem" }}>
                  <div style={{ display: "grid", gridTemplateColumns: "2.5rem 1fr 5rem 1fr", gap: "1rem", alignItems: "center" }}>
                    <span className="small-caps" style={{ fontSize: "0.625rem", color: "var(--muted-foreground)" }}>Rank</span>
                    <span className="small-caps" style={{ fontSize: "0.625rem", color: "var(--muted-foreground)" }}>Feature</span>
                    <span className="small-caps" style={{ fontSize: "0.625rem", color: "var(--muted-foreground)", textAlign: "center" }}>Value</span>
                    <span className="small-caps" style={{ fontSize: "0.625rem", color: "var(--muted-foreground)" }}>Importance</span>
                  </div>
                </div>
                <hr className="rule" />
                <div style={{ padding: "0 2rem 1.5rem" }}>
                  {results.worst_parameters.slice(0, 10).map((param, index) => {
                    const maxImp = results.worst_parameters[0]?.importance || 1;
                    const barWidth = (param.importance / maxImp) * 100;

                    return (
                      <div
                        key={param.feature}
                        style={{
                          display: "grid",
                          gridTemplateColumns: "2.5rem 1fr 5rem 1fr",
                          gap: "1rem",
                          alignItems: "center",
                          padding: "1rem 0",
                          borderBottom: index < 9 ? "1px solid var(--border)" : "none",
                        }}
                      >
                        <span className="param-rank">{index + 1}</span>
                        <div>
                          <div className="param-name">{param.feature}</div>
                          <div className="param-desc">{param.description}</div>
                        </div>
                        <div className="param-value">{param.input_value.toLocaleString()}</div>
                        <div>
                          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
                            <div className="importance-bar-track" style={{ flex: 1 }}>
                              <div
                                className="importance-bar-fill"
                                style={{ width: `${barWidth}%` }}
                              />
                            </div>
                            <span style={{
                              fontFamily: "var(--font-mono)",
                              fontSize: "0.75rem",
                              color: "var(--muted-foreground)",
                              minWidth: "3.5rem",
                              textAlign: "right",
                            }}>
                              {(param.importance * 100).toFixed(1)}%
                            </span>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          </section>

          {/* ── Individual Model Results ── */}
          <section style={{ padding: "3rem 0 5rem" }} className="fade-in stagger-3">
            <div className="container-narrow">
              <SectionLabel text="Individual Model Results" />

              <div className="grid-3">
                {results.model_results
                  .filter((r) => !r.error)
                  .sort((a, b) => b.probability_default - a.probability_default)
                  .map((model) => {
                    const flagClass = model.risk_flag.toLowerCase();

                    return (
                      <div
                        className={`card model-card model-card-${flagClass}`}
                        key={model.model_name}
                      >
                        <div className="model-card-header">
                          <div>
                            <div className="small-caps" style={{ fontSize: "0.6875rem", marginBottom: "0.25rem", color: "var(--muted-foreground)" }}>
                              {model.model_name}
                            </div>
                            <span className={`badge badge-${flagClass}`}>
                              <span style={{ color: getRiskColor(model.risk_flag), fontSize: "0.5rem" }}>
                                {getRiskEmoji(model.risk_flag)}
                              </span>
                              {model.risk_flag}
                            </span>
                          </div>
                          <div className="model-card-prob">
                            {(model.probability_default * 100).toFixed(1)}
                            <span style={{ fontSize: "0.875rem", opacity: 0.6 }}>%</span>
                          </div>
                        </div>

                        {/* Probability bar */}
                        <div className="progress-track" style={{ marginBottom: "1rem" }}>
                          <div
                            className={`progress-fill progress-fill-${flagClass}`}
                            style={{ width: `${model.probability_default * 100}%` }}
                          />
                        </div>

                        {/* Prediction */}
                        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                          <span style={{ fontSize: "0.8125rem", color: "var(--muted-foreground)" }}>
                            Prediction
                          </span>
                          <span style={{
                            fontFamily: "var(--font-mono)",
                            fontSize: "0.75rem",
                            fontWeight: 600,
                            letterSpacing: "0.05em",
                            color: model.prediction === 1 ? "#a02020" : "#1a7a1a",
                          }}>
                            {model.prediction_label}
                          </span>
                        </div>

                        {/* Top 3 important features for this model */}
                        {model.feature_importance && (
                          <>
                            <hr className="rule" style={{ margin: "1rem 0 0.75rem" }} />
                            <div className="small-caps" style={{ fontSize: "0.5625rem", color: "var(--muted-foreground)", marginBottom: "0.5rem" }}>
                              Top Contributing Features
                            </div>
                            {Object.entries(model.feature_importance)
                              .sort(([, a], [, b]) => b - a)
                              .slice(0, 3)
                              .map(([feat, imp]) => (
                                <div
                                  key={feat}
                                  style={{
                                    display: "flex",
                                    justifyContent: "space-between",
                                    alignItems: "center",
                                    fontSize: "0.8125rem",
                                    padding: "0.25rem 0",
                                  }}
                                >
                                  <span style={{ color: "var(--muted-foreground)" }}>
                                    {results.feature_descriptions?.[feat] || feat}
                                  </span>
                                  <span style={{
                                    fontFamily: "var(--font-mono)",
                                    fontSize: "0.6875rem",
                                    color: "var(--accent)",
                                  }}>
                                    {(imp * 100).toFixed(1)}%
                                  </span>
                                </div>
                              ))}
                          </>
                        )}
                      </div>
                    );
                  })}
              </div>

              {/* Models with errors */}
              {results.model_results.filter((r) => r.error).length > 0 && (
                <div style={{ marginTop: "2rem" }}>
                  <div className="small-caps" style={{ color: "var(--muted-foreground)", marginBottom: "0.75rem" }}>
                    Models with Errors
                  </div>
                  {results.model_results
                    .filter((r) => r.error)
                    .map((model) => (
                      <div
                        key={model.model_name}
                        style={{
                          padding: "0.75rem 1rem",
                          fontSize: "0.875rem",
                          color: "#a02020",
                          background: "rgba(178,34,34,0.04)",
                          borderRadius: "var(--radius-md)",
                          marginBottom: "0.5rem",
                        }}
                      >
                        <strong>{model.model_name}:</strong> {model.error}
                      </div>
                    ))}
                </div>
              )}
            </div>
          </section>
        </div>
      )}

      {/* ── Footer ── */}
      <footer className="footer">
        <div className="container-narrow">
          <p style={{ fontFamily: "var(--font-mono)", fontSize: "0.75rem", letterSpacing: "0.08em", textTransform: "uppercase" }}>
            Credit Risk Analysis · 15-Model ML Ensemble · UCI Credit Card Dataset
          </p>
        </div>
      </footer>
    </main>
  );
}
