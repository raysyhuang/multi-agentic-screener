I'll start by reading through your repos and the key references to understand the landscape, then give you a comprehensive brainstorm.Now let me read the frontend design skill since you want a clean JS frontend, then I'll put together the comprehensive brainstorm document.Now let me create a comprehensive, beautifully designed brainstorm document as an interactive React artifact that maps out the entire architecture, decisions, and actionable plan.Here's your full brainstorm packed into an interactive dashboard. Click through the 10 tabs to explore everything. Let me highlight the most important takeaways:

**The core recommendation is a hybrid approach**: build a standalone Python backend (LangGraph orchestration + OpenBB data layer + your API keys) that runs on autopilot every morning at 6 AM ET, while keeping Claude Code / Cursor for deep-dive interactive sessions when you want to dig into a particular candidate.

**The key architecture insight from TradingAgents** (30k stars, research-backed) is the bull/bear debate mechanism — having opposing agents argue for and against each trade forces cross-validation and kills weak theses. This is the single biggest upgrade over single-agent screeners like your current KooCore-D.

**Capabilities you're probably not thinking about yet**: regime detection (your screening criteria should change based on whether we're in a bull/bear/chop market), chain-of-thought verification (a dedicated agent that checks other agents for hallucinations), and multi-model ensemble voting (run the same analysis through Claude + GPT + Gemini independently, only trade when all three agree).

**The biggest risk to be honest about**: 5-10% per trade compounded 7-100x is extremely aggressive. Even world-class systematic traders average 1-3% per trade. I'd recommend tracking actual performance from day one and letting the data tell you what's realistic, rather than targeting a specific compound return.

The roadmap gives you a working v1 in about 4 weeks, with self-improvement and advanced features rolling in through month 3.

import { useState } from "react";

const sections = [
  {
    id: "verdict",
    title: "The Big Decision",
    icon: "⚖️",
    color: "#e8b931",
  },
  {
    id: "architecture",
    title: "System Architecture",
    icon: "🏗️",
    color: "#4ecdc4",
  },
  {
    id: "agents",
    title: "Agent Roles & LLM Assignments",
    icon: "🤖",
    color: "#ff6b6b",
  },
  {
    id: "pipeline",
    title: "Daily Pipeline",
    icon: "⚙️",
    color: "#a78bfa",
  },
  {
    id: "data",
    title: "Data Sources & APIs",
    icon: "📊",
    color: "#34d399",
  },
  {
    id: "capabilities",
    title: "Hidden Capabilities You're Missing",
    icon: "💡",
    color: "#f472b6",
  },
  {
    id: "frontend",
    title: "Frontend Strategy",
    icon: "🎨",
    color: "#60a5fa",
  },
  {
    id: "repos",
    title: "What to Steal from Each Repo",
    icon: "📦",
    color: "#fb923c",
  },
  {
    id: "risks",
    title: "Risks & Reality Check",
    icon: "⚠️",
    color: "#ef4444",
  },
  {
    id: "roadmap",
    title: "Build Roadmap",
    icon: "🗺️",
    color: "#8b5cf6",
  },
];

const content = {
  verdict: {
    recommendation: "Build a Hybrid",
    subtitle:
      "A standalone Python backend with multi-agent orchestration + a clean React frontend, but designed to also be runnable via Claude Code / Cursor for interactive analysis sessions.",
    options: [
      {
        name: "Option A: Full Standalone App",
        pros: [
          "Runs unattended on a schedule (cron / Heroku / Railway)",
          "You own the full stack — no dependency on Claude.ai sessions",
          "Can evolve into a SaaS product",
          "Telegram/Slack alerts built-in",
        ],
        cons: [
          "More upfront build effort",
          "You manage infra, API costs, error recovery",
          "No real-time 'thinking out loud' like Claude Code gives you",
        ],
        verdict: "Best for daily automation",
      },
      {
        name: "Option B: Claude Code / Cursor Orchestrator",
        pros: [
          "Near-zero infra — runs on your machine",
          "Already has web search, MCPs, file access, code execution",
          "You can talk to it, ask follow-ups, refine in real time",
          "Extremely fast to prototype",
        ],
        cons: [
          "Requires you to be present to trigger it (not truly automated)",
          "Session-based — no persistent state between runs",
          "API costs can spike with long agent chains",
          "Hard to backtest systematically",
        ],
        verdict: "Best for interactive research",
      },
      {
        name: "Option C: Hybrid (Recommended)",
        pros: [
          "Standalone backend runs daily on autopilot",
          "Claude Code / Cursor used for deep-dive sessions on candidates",
          "React dashboard shows results, history, and performance tracking",
          "Best of both worlds — automation + human-in-the-loop",
        ],
        cons: [
          "Two systems to maintain",
          "Need to define clear handoff points between automated and interactive",
        ],
        verdict: "★ RECOMMENDED — Gets you automation + flexibility",
      },
    ],
  },
  architecture: {
    layers: [
      {
        name: "Layer 1 — Data Ingestion",
        description:
          "Parallel fetchers for price data, fundamentals, news, sentiment, insider transactions, options flow, SEC filings. Uses your Polygon, FMP, Financial Datasets, and free sources (yfinance, SEC EDGAR).",
        tech: "Python asyncio + aiohttp, OpenBB SDK as unified data layer",
      },
      {
        name: "Layer 2 — Agent Orchestration",
        description:
          "Multi-agent graph where specialized agents (scanner, analyst, validator, predictor, risk manager) communicate via structured messages. Agents debate and cross-validate before producing a final signal.",
        tech: "LangGraph (used by TradingAgents) or CrewAI or AutoGen. LangGraph recommended — most mature for financial workflows, supports cycles/debates.",
      },
      {
        name: "Layer 3 — LLM Router",
        description:
          "Routes tasks to the right model: cheap/fast models for data retrieval and formatting, expensive/deep models for analysis and reasoning. Saves 60-80% on API costs.",
        tech: "Custom router: Claude Haiku 4.5 for formatting → Claude Opus 4.6 or GPT-o3 for analysis → Gemini for broad web research",
      },
      {
        name: "Layer 4 — Signal Generation & Backtesting",
        description:
          "Converts agent consensus into structured trade signals (ticker, direction, entry zone, stop loss, target, confidence score, time horizon). Backtests against historical data.",
        tech: "backtrader or vectorbt for backtesting, custom signal schema",
      },
      {
        name: "Layer 5 — Frontend & Alerts",
        description:
          "React dashboard for viewing daily picks, historical performance, agent reasoning traces. Telegram bot for real-time alerts.",
        tech: "React + Vite + TailwindCSS, Telegram Bot API, optional: WebSocket for live updates",
      },
      {
        name: "Layer 6 — Persistence & Learning",
        description:
          "SQLite/Postgres stores every prediction, actual outcome, and agent reasoning. Over time, this becomes training data for improving prompts and identifying which agent patterns correlate with wins.",
        tech: "SQLAlchemy + PostgreSQL (or SQLite for local), structured logging",
      },
    ],
  },
  agents: {
    roles: [
      {
        name: "🔍 Universe Scanner",
        llm: "Claude Haiku 4.5 (fast, cheap)",
        job: "Scans full NYSE/NASDAQ universe daily. Filters by: volume surge (>1.5x 20-day avg), price near key moving averages, RSI in 30-50 zone (pre-breakout), unusual options activity, recent earnings surprise. Outputs 20-50 candidates.",
        tools: "Polygon API, FinViz screener, OpenBB equity screener",
      },
      {
        name: "📈 Technical Analyst",
        llm: "Claude Sonnet 4.5 (balanced)",
        job: "For each candidate: chart pattern recognition, support/resistance levels, volume profile analysis, multi-timeframe alignment (daily + weekly + 4h), momentum indicators. Outputs a technical score 1-10 with reasoning.",
        tools: "ta-lib, pandas-ta, custom chart generation",
      },
      {
        name: "📊 Fundamental Analyst",
        llm: "Claude Opus 4.6 (deep reasoning)",
        job: "Earnings quality, revenue acceleration, margin expansion, insider buying patterns, institutional flow, debt ratios, competitive positioning. For short-term trades, focuses on catalysts: upcoming earnings, FDA decisions, product launches.",
        tools: "FMP API, SEC EDGAR, Financial Datasets API",
      },
      {
        name: "📰 Sentiment & News Analyst",
        llm: "Gemini 2.5 Pro (strong at web research)",
        job: "Real-time news scanning, social media sentiment (Reddit, X/Twitter, StockTwits), analyst upgrades/downgrades, short interest changes. Identifies narrative momentum — is the 'story' building or fading?",
        tools: "Web search, news APIs, social media APIs",
      },
      {
        name: "🐂 Bull Researcher",
        llm: "Claude Opus 4.6",
        job: "Synthesizes analyst reports into a bullish thesis. Actively argues FOR the trade. Must ground arguments in specific data points.",
        tools: "Reads outputs from all analysts",
      },
      {
        name: "🐻 Bear Researcher",
        llm: "GPT-o3 (strong adversarial reasoning)",
        job: "Actively argues AGAINST the trade. Identifies risks, headwinds, bearish patterns, sector weakness, macro threats. Tries to kill the thesis.",
        tools: "Reads outputs from all analysts + contrarian data",
      },
      {
        name: "🎯 Trade Signal Generator",
        llm: "Claude Opus 4.6",
        job: "Reads the bull/bear debate transcript. Weighs evidence. Produces a final signal with: entry price, stop loss, target(s), position size suggestion, confidence score (0-100), and time horizon.",
        tools: "Structured output schema, risk calculators",
      },
      {
        name: "🛡️ Risk Manager",
        llm: "Claude Opus 4.6",
        job: "Reviews the final signal against: portfolio concentration, correlation with existing positions, max drawdown tolerance, sector exposure, overall market regime (bull/bear/chop). Can VETO a trade.",
        tools: "Portfolio state, VIX data, correlation matrices",
      },
      {
        name: "📝 Meta-Analyst (Self-Improvement)",
        llm: "Claude Opus 4.6 (weekly, not daily)",
        job: "Reviews past predictions vs actual outcomes. Identifies systematic biases (e.g., 'our technical analyst is too bullish on RSI signals'). Suggests prompt improvements. This is how the system learns.",
        tools: "Prediction database, performance metrics",
      },
    ],
  },
  pipeline: {
    steps: [
      {
        time: "6:00 AM ET (Pre-Market)",
        action: "Universe Scan",
        detail:
          "Scanner agent filters 8,000+ tickers → 20-50 candidates based on volume, price action, news catalysts. Takes ~2 minutes with parallel API calls.",
      },
      {
        time: "6:05 AM",
        action: "Parallel Analysis",
        detail:
          "Technical, Fundamental, and Sentiment agents analyze all candidates simultaneously. Each produces a scored report. Takes ~5-8 minutes.",
      },
      {
        time: "6:15 AM",
        action: "Top-10 Filter",
        detail:
          "Candidates ranked by composite score. Top 10 proceed to deep analysis.",
      },
      {
        time: "6:20 AM",
        action: "Bull/Bear Debate",
        detail:
          "For each top-10 candidate, Bull and Bear researchers argue 2-3 rounds. This is where cross-validation happens — weak theses get exposed.",
      },
      {
        time: "6:35 AM",
        action: "Signal Generation",
        detail:
          "Trade Signal Generator reads debates, produces structured signals for top 1-3 candidates.",
      },
      {
        time: "6:40 AM",
        action: "Risk Check & Final Output",
        detail:
          "Risk Manager reviews signals against portfolio state. Approved signals get sent to: (1) Dashboard, (2) Telegram alert, (3) Database for tracking.",
      },
      {
        time: "4:30 PM ET (Post-Market)",
        action: "Outcome Recording",
        detail:
          "System records what actually happened to each pick. Did it hit the target? Stop loss? Still open?",
      },
      {
        time: "Sunday Weekly",
        action: "Meta-Analysis & Self-Improvement",
        detail:
          "Meta-Analyst reviews weekly performance. Adjusts agent prompts, scoring weights, and screening criteria based on what worked and what didn't.",
      },
    ],
  },
  data: {
    sources: [
      {
        name: "Polygon.io",
        key: "POLYGON_API_KEY",
        use: "Real-time & historical price data, options data, news, financials. Your primary market data source.",
        tier: "Core",
      },
      {
        name: "FMP (Financial Modeling Prep)",
        key: "FMP_API_KEY",
        use: "Fundamentals, financial statements, earnings calendar, analyst estimates, insider transactions, institutional holdings.",
        tier: "Core",
      },
      {
        name: "Financial Datasets",
        key: "FINANCIAL_DATASETS_API_KEY",
        use: "Alternative datasets, specialized financial data. Check what's available on their API.",
        tier: "Core",
      },
      {
        name: "OpenBB Platform",
        key: "Uses above keys",
        use: "Unified API layer that wraps 90+ data providers into consistent endpoints. Huge time saver — use as your data abstraction layer.",
        tier: "Recommended",
      },
      {
        name: "SEC EDGAR",
        key: "Free",
        use: "10-K, 10-Q, 8-K filings. Insider transactions (Form 4). Institutional holdings (13-F). Critical for fundamental analysis.",
        tier: "Free",
      },
      {
        name: "yfinance",
        key: "Free",
        use: "Backup price data, quick screening. Rate-limited but great for prototyping.",
        tier: "Free",
      },
      {
        name: "FRED (Federal Reserve)",
        key: "Free",
        use: "Macro indicators: interest rates, GDP, unemployment, yield curve. Essential for market regime detection.",
        tier: "Free",
      },
      {
        name: "Reddit/StockTwits APIs",
        key: "Various",
        use: "Retail sentiment. Trending tickers. Meme stock detection (to avoid or ride).",
        tier: "Supplementary",
      },
      {
        name: "Unusual Whales / CBOE",
        key: "Paid",
        use: "Options flow data — large institutional bets. Smart money tracking. Very powerful for short-term signals.",
        tier: "Upgrade later",
      },
    ],
  },
  capabilities: {
    items: [
      {
        name: "Agent Memory & Reflection",
        description:
          "Agents that remember past predictions and learn from mistakes. Not just logging — actual prompt modification based on performance patterns. FinMem (from the literature) uses layered memory: working memory (current session), episodic memory (past trades), and semantic memory (learned rules).",
        impact: "High",
      },
      {
        name: "Chain-of-Thought Verification",
        description:
          "One agent generates reasoning, another agent explicitly checks each step for logical errors, hallucinations, or unsupported claims. This catches the #1 failure mode of LLM trading agents: confident but wrong analysis.",
        impact: "Critical",
      },
      {
        name: "Regime Detection Agent",
        description:
          "A dedicated agent that classifies the current market as: trending-up, trending-down, range-bound, high-volatility, or crisis. Your screening criteria should CHANGE based on regime. In a bear market, scanning for breakouts is foolish.",
        impact: "High",
      },
      {
        name: "Adversarial Red-Teaming",
        description:
          "Beyond bull/bear debate: an agent specifically tries to find flaws in the SYSTEM's reasoning, not just the trade thesis. 'Are we overfitting to recent patterns?' 'Is our sentiment agent just echoing hype?'",
        impact: "Medium",
      },
      {
        name: "Multi-Model Ensemble Voting",
        description:
          "Run the same analysis through Claude, GPT, and Gemini independently, then vote. When all three agree, confidence is much higher. When they disagree, that's a signal to skip the trade.",
        impact: "High",
      },
      {
        name: "Tool Use & Code Execution",
        description:
          "Agents can write and execute Python code on-the-fly for custom analysis: calculate correlations, run quick backtests, generate charts, perform statistical tests. This is what makes LLM agents fundamentally different from rule-based screeners.",
        impact: "Critical",
      },
      {
        name: "Web Browsing & Real-Time Research",
        description:
          "Agents with web access can check breaking news, read SEC filings in real-time, check competitor earnings, look up management changes — things no static screener can do.",
        impact: "High",
      },
      {
        name: "Structured Output Enforcement",
        description:
          "Force agents to output in strict JSON schemas. This prevents rambling, ensures every signal has stop loss/targets/confidence, and makes downstream processing reliable.",
        impact: "Critical",
      },
      {
        name: "Prompt Evolution via Genetic Algorithms",
        description:
          "After enough prediction data: treat agent prompts as 'genes', mutate them, and select for prompts that produce better predictions. Automated prompt optimization.",
        impact: "Long-term",
      },
      {
        name: "Synthetic Data Generation for Backtesting",
        description:
          "Use LLMs to generate synthetic market scenarios (flash crash, sector rotation, earnings surprise) to stress-test the system without waiting for real events.",
        impact: "Medium",
      },
    ],
  },
  frontend: {
    principles: [
      "React + Vite + Tailwind — fast, modern, no Streamlit",
      "shadcn/ui components for clean, professional look",
      "Dark theme by default (traders prefer it)",
      "Mobile-responsive (check picks on your phone)",
      "WebSocket connection for real-time agent progress",
    ],
    pages: [
      {
        name: "Daily Dashboard",
        description:
          "Today's 1-3 picks with confidence meters, entry/exit zones, agent reasoning summaries. Click to expand full bull/bear debate transcripts.",
      },
      {
        name: "Pipeline Monitor",
        description:
          "Real-time view of the daily pipeline: which agents are running, their progress, any errors. Like a CI/CD dashboard for your trading pipeline.",
      },
      {
        name: "Historical Performance",
        description:
          "Track record: win rate, average return, max drawdown, Sharpe ratio. Charts showing cumulative performance. Filter by time period, confidence level, market regime.",
      },
      {
        name: "Agent Insights",
        description:
          "Which agents are most accurate? Which have systematic biases? Performance breakdown per agent over time. This is your self-improvement dashboard.",
      },
      {
        name: "Candidate Explorer",
        description:
          "Browse all 20-50 daily candidates, not just the top picks. See why they were filtered out. Sometimes great trades are in the 'almost' pile.",
      },
      {
        name: "Settings & Configuration",
        description:
          "Adjust screening criteria, agent LLM assignments, risk parameters, alert preferences. No code changes needed for tuning.",
      },
    ],
  },
  repos: {
    items: [
      {
        name: "TradingAgents (30k stars)",
        steal:
          "The entire multi-agent architecture pattern. Bull/Bear debate mechanism. LangGraph integration. Multi-LLM router pattern. Their agent role definitions are research-backed and proven.",
        link: "github.com/TauricResearch/TradingAgents",
      },
      {
        name: "OpenBB Platform",
        steal:
          "Use as your unified data layer. Don't build 10 different API integrations — OpenBB wraps 90+ providers. Their equity screener, fundamental data standardization, and MCP server integration are production-ready.",
        link: "github.com/OpenBB-finance/OpenBB",
      },
      {
        name: "AgenticTrading",
        steal:
          "Their orchestration framework for financial agents. The 'From Algorithmic Trading to Agentic Trading' paradigm. Good patterns for converting agent outputs into executable signals.",
        link: "github.com/Open-Finance-Lab/AgenticTrading",
      },
      {
        name: "FinRobot",
        steal:
          "Their agent platform architecture for financial analysis. Good patterns for LLM-powered fundamental analysis and report generation.",
        link: "github.com/AI4Finance-Foundation/FinRobot",
      },
      {
        name: "FinGPT",
        steal:
          "Sentiment analysis fine-tuning approaches. Their financial NLP pipeline. Good baseline for what sentiment signals actually move prices.",
        link: "github.com/AI4Finance-Foundation/FinGPT",
      },
      {
        name: "Qlib (Microsoft)",
        steal:
          "Production-grade quant research framework. Their alpha factor library, backtesting engine, and model serving infrastructure. The gold standard for systematic trading research.",
        link: "github.com/microsoft/qlib",
      },
      {
        name: "Your KooCore-D",
        steal:
          "Your existing screening logic — don't throw it away. Wrap your proven filters as a tool that agents can call. Your domain knowledge encoded in code is valuable.",
        link: "github.com/raysyhuang/KooCore-D",
      },
    ],
  },
  risks: {
    items: [
      {
        risk: "5-10% per trade compounded 7-100x is extremely aggressive",
        reality:
          "Even the best systematic traders average 1-3% per trade with 55-60% win rates. A 5% average return per trade compounded 20 times in a year would be exceptional. Set realistic baselines: track your ACTUAL win rate and average return, then optimize from there.",
        severity: "Critical",
      },
      {
        risk: "LLM hallucination in financial analysis",
        reality:
          "LLMs confidently state incorrect financial figures, invent patterns that don't exist, and misinterpret data. Mitigation: always ground agent outputs in verifiable data (actual numbers from APIs, not LLM memory). The verification agent is not optional.",
        severity: "Critical",
      },
      {
        risk: "API costs can spiral",
        reality:
          "Running 9 agents with Opus-level models on 50 candidates daily could cost $20-50/day or more. The LLM router (cheap models for cheap tasks) is essential. Monitor costs weekly.",
        severity: "High",
      },
      {
        risk: "Overfitting to recent market regime",
        reality:
          "A system built in a bull market will fail in a bear market. The regime detection agent and the meta-analyst reviewing performance across regimes are your insurance.",
        severity: "High",
      },
      {
        risk: "Survivorship bias in backtesting",
        reality:
          "You can only backtest LLM agents on data the LLMs haven't seen (post-training cutoff). Historical backtests of LLM reasoning are fundamentally limited. Forward-test with paper trading first.",
        severity: "Medium",
      },
      {
        risk: "Latency — agents are slow",
        reality:
          "A full 9-agent pipeline on 50 candidates will take 15-40 minutes. That's fine for a pre-market screener, but don't expect real-time execution. This is a screener, not an execution system.",
        severity: "Low",
      },
    ],
  },
  roadmap: {
    phases: [
      {
        phase: "Phase 1 — Foundation (Weeks 1-2)",
        tasks: [
          "Set up project structure with Python backend + React frontend",
          "Integrate OpenBB as unified data layer",
          "Build the Universe Scanner agent with basic filters",
          "Implement LLM router (Haiku for fast tasks, Opus for deep analysis)",
          "Create the structured signal schema (JSON output format)",
          "Set up SQLite database for storing predictions",
          "Deploy basic Telegram alerts",
        ],
      },
      {
        phase: "Phase 2 — Core Agents (Weeks 3-4)",
        tasks: [
          "Build Technical, Fundamental, and Sentiment analyst agents",
          "Implement Bull/Bear researcher debate (borrow from TradingAgents pattern)",
          "Build Trade Signal Generator with structured outputs",
          "Build Risk Manager with veto capability",
          "Create agent orchestration with LangGraph",
          "End-to-end pipeline test on 5 tickers",
        ],
      },
      {
        phase: "Phase 3 — Frontend & Tracking (Weeks 5-6)",
        tasks: [
          "Build React dashboard with daily picks view",
          "Historical performance tracking and charting",
          "Agent reasoning trace viewer (expandable debate transcripts)",
          "Pipeline monitoring view",
          "Mobile-responsive design",
          "Settings page for tuning parameters",
        ],
      },
      {
        phase: "Phase 4 — Self-Improvement & Production (Weeks 7-8)",
        tasks: [
          "Build Meta-Analyst agent for weekly self-review",
          "Implement multi-model ensemble voting",
          "Add regime detection agent",
          "Paper trading mode with automated outcome tracking",
          "Deploy to Railway/Fly.io or local cron job",
          "Cost monitoring and optimization",
        ],
      },
      {
        phase: "Phase 5 — Advanced (Month 3+)",
        tasks: [
          "Prompt evolution based on prediction performance",
          "Options flow integration (Unusual Whales)",
          "Sector rotation detection",
          "Correlation analysis with existing portfolio",
          "Claude Code integration for deep-dive interactive sessions",
          "Backtest framework for systematic evaluation",
        ],
      },
    ],
  },
};

// Color helpers
const severity = { Critical: "#ef4444", High: "#f59e0b", Medium: "#3b82f6", Low: "#6b7280" };
const impactColors = {
  Critical: "#ef4444",
  High: "#f59e0b",
  Medium: "#3b82f6",
  "Long-term": "#8b5cf6",
};
const tierColors = {
  Core: "#ef4444",
  Recommended: "#f59e0b",
  Free: "#34d399",
  Supplementary: "#3b82f6",
  "Upgrade later": "#8b5cf6",
};

export default function BrainstormDashboard() {
  const [active, setActive] = useState("verdict");
  const [expandedOption, setExpandedOption] = useState(null);
  const [expandedAgent, setExpandedAgent] = useState(null);
  const [expandedCapability, setExpandedCapability] = useState(null);

  const renderVerdict = () => {
    const d = content.verdict;
    return (
      <div>
        <div
          style={{
            background: "linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)",
            borderRadius: 16,
            padding: 32,
            marginBottom: 24,
            border: "1px solid #e8b93133",
          }}
        >
          <div
            style={{
              fontSize: 14,
              color: "#e8b931",
              textTransform: "uppercase",
              letterSpacing: 2,
              marginBottom: 8,
            }}
          >
            Recommendation
          </div>
          <h2
            style={{
              fontSize: 32,
              fontWeight: 800,
              color: "#e8b931",
              margin: "0 0 12px",
            }}
          >
            {d.recommendation}
          </h2>
          <p style={{ color: "#94a3b8", fontSize: 16, lineHeight: 1.7, margin: 0 }}>
            {d.subtitle}
          </p>
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          {d.options.map((opt, i) => {
            const isExpanded = expandedOption === i;
            const isRecommended = i === 2;
            return (
              <div
                key={i}
                onClick={() => setExpandedOption(isExpanded ? null : i)}
                style={{
                  background: isRecommended
                    ? "linear-gradient(135deg, #1a2e1a 0%, #162e16 100%)"
                    : "#111827",
                  borderRadius: 12,
                  padding: 20,
                  cursor: "pointer",
                  border: isRecommended
                    ? "1px solid #34d39955"
                    : "1px solid #1e293b",
                  transition: "all 0.2s",
                }}
              >
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                  }}
                >
                  <h3
                    style={{
                      color: isRecommended ? "#34d399" : "#e2e8f0",
                      fontSize: 18,
                      fontWeight: 700,
                      margin: 0,
                    }}
                  >
                    {opt.name}
                  </h3>
                  <span
                    style={{
                      fontSize: 12,
                      padding: "4px 12px",
                      borderRadius: 20,
                      background: isRecommended ? "#34d39922" : "#1e293b",
                      color: isRecommended ? "#34d399" : "#94a3b8",
                    }}
                  >
                    {opt.verdict}
                  </span>
                </div>
                {isExpanded && (
                  <div style={{ marginTop: 16, display: "flex", gap: 24, flexWrap: "wrap" }}>
                    <div style={{ flex: 1, minWidth: 200 }}>
                      <div
                        style={{
                          fontSize: 12,
                          color: "#34d399",
                          fontWeight: 600,
                          marginBottom: 8,
                          textTransform: "uppercase",
                          letterSpacing: 1,
                        }}
                      >
                        Pros
                      </div>
                      {opt.pros.map((p, j) => (
                        <div
                          key={j}
                          style={{
                            color: "#cbd5e1",
                            fontSize: 14,
                            padding: "4px 0",
                            paddingLeft: 12,
                            borderLeft: "2px solid #34d39944",
                            marginBottom: 6,
                          }}
                        >
                          {p}
                        </div>
                      ))}
                    </div>
                    <div style={{ flex: 1, minWidth: 200 }}>
                      <div
                        style={{
                          fontSize: 12,
                          color: "#ef4444",
                          fontWeight: 600,
                          marginBottom: 8,
                          textTransform: "uppercase",
                          letterSpacing: 1,
                        }}
                      >
                        Cons
                      </div>
                      {opt.cons.map((c, j) => (
                        <div
                          key={j}
                          style={{
                            color: "#cbd5e1",
                            fontSize: 14,
                            padding: "4px 0",
                            paddingLeft: 12,
                            borderLeft: "2px solid #ef444444",
                            marginBottom: 6,
                          }}
                        >
                          {c}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  const renderArchitecture = () => (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      {content.architecture.layers.map((layer, i) => (
        <div
          key={i}
          style={{
            background: "#111827",
            borderRadius: 12,
            padding: 20,
            borderLeft: `4px solid ${
              ["#ef4444", "#f59e0b", "#34d399", "#3b82f6", "#a78bfa", "#f472b6"][i]
            }`,
          }}
        >
          <h3 style={{ color: "#e2e8f0", fontSize: 16, fontWeight: 700, margin: "0 0 8px" }}>
            {layer.name}
          </h3>
          <p style={{ color: "#94a3b8", fontSize: 14, margin: "0 0 12px", lineHeight: 1.6 }}>
            {layer.description}
          </p>
          <div
            style={{
              background: "#0d1117",
              borderRadius: 8,
              padding: "8px 14px",
              fontSize: 13,
              color: "#7dd3fc",
              fontFamily: "monospace",
            }}
          >
            {layer.tech}
          </div>
        </div>
      ))}
    </div>
  );

  const renderAgents = () => (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      {content.agents.roles.map((agent, i) => {
        const isExpanded = expandedAgent === i;
        return (
          <div
            key={i}
            onClick={() => setExpandedAgent(isExpanded ? null : i)}
            style={{
              background: "#111827",
              borderRadius: 12,
              padding: 16,
              cursor: "pointer",
              border: "1px solid #1e293b",
              transition: "all 0.2s",
            }}
          >
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
              }}
            >
              <h3 style={{ color: "#e2e8f0", fontSize: 16, fontWeight: 700, margin: 0 }}>
                {agent.name}
              </h3>
              <span
                style={{
                  fontSize: 11,
                  padding: "3px 10px",
                  borderRadius: 20,
                  background: "#1e293b",
                  color: "#7dd3fc",
                  fontFamily: "monospace",
                  whiteSpace: "nowrap",
                }}
              >
                {agent.llm}
              </span>
            </div>
            {isExpanded && (
              <div style={{ marginTop: 12 }}>
                <p style={{ color: "#94a3b8", fontSize: 14, lineHeight: 1.7, margin: "0 0 12px" }}>
                  {agent.job}
                </p>
                <div
                  style={{
                    background: "#0d1117",
                    borderRadius: 8,
                    padding: "8px 14px",
                    fontSize: 13,
                    color: "#a78bfa",
                    fontFamily: "monospace",
                  }}
                >
                  Tools: {agent.tools}
                </div>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );

  const renderPipeline = () => (
    <div style={{ position: "relative" }}>
      <div
        style={{
          position: "absolute",
          left: 89,
          top: 24,
          bottom: 24,
          width: 2,
          background: "linear-gradient(180deg, #a78bfa 0%, #a78bfa44 100%)",
        }}
      />
      <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
        {content.pipeline.steps.map((step, i) => (
          <div key={i} style={{ display: "flex", gap: 20, alignItems: "flex-start" }}>
            <div
              style={{
                minWidth: 80,
                fontSize: 12,
                color: "#a78bfa",
                fontFamily: "monospace",
                textAlign: "right",
                paddingTop: 2,
                fontWeight: 600,
              }}
            >
              {step.time}
            </div>
            <div
              style={{
                width: 12,
                height: 12,
                borderRadius: "50%",
                background: "#a78bfa",
                marginTop: 4,
                flexShrink: 0,
                zIndex: 1,
              }}
            />
            <div
              style={{
                background: "#111827",
                borderRadius: 12,
                padding: 16,
                flex: 1,
                border: "1px solid #1e293b",
              }}
            >
              <h3 style={{ color: "#e2e8f0", fontSize: 15, fontWeight: 700, margin: "0 0 6px" }}>
                {step.action}
              </h3>
              <p style={{ color: "#94a3b8", fontSize: 13, margin: 0, lineHeight: 1.6 }}>
                {step.detail}
              </p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  const renderData = () => (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      {content.data.sources.map((src, i) => (
        <div
          key={i}
          style={{
            background: "#111827",
            borderRadius: 12,
            padding: 16,
            border: "1px solid #1e293b",
            display: "flex",
            gap: 16,
            alignItems: "flex-start",
          }}
        >
          <div style={{ flex: 1 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
              <h3 style={{ color: "#e2e8f0", fontSize: 15, fontWeight: 700, margin: 0 }}>
                {src.name}
              </h3>
              <span
                style={{
                  fontSize: 10,
                  padding: "2px 8px",
                  borderRadius: 20,
                  background: tierColors[src.tier] + "22",
                  color: tierColors[src.tier],
                  fontWeight: 600,
                  textTransform: "uppercase",
                  letterSpacing: 1,
                }}
              >
                {src.tier}
              </span>
            </div>
            <p style={{ color: "#94a3b8", fontSize: 13, margin: 0, lineHeight: 1.6 }}>
              {src.use}
            </p>
          </div>
          <div
            style={{
              fontSize: 11,
              fontFamily: "monospace",
              color: "#6b7280",
              whiteSpace: "nowrap",
            }}
          >
            {src.key}
          </div>
        </div>
      ))}
    </div>
  );

  const renderCapabilities = () => (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      {content.capabilities.items.map((cap, i) => {
        const isExpanded = expandedCapability === i;
        return (
          <div
            key={i}
            onClick={() => setExpandedCapability(isExpanded ? null : i)}
            style={{
              background: "#111827",
              borderRadius: 12,
              padding: 16,
              cursor: "pointer",
              border: "1px solid #1e293b",
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <h3 style={{ color: "#e2e8f0", fontSize: 15, fontWeight: 700, margin: 0 }}>
                {cap.name}
              </h3>
              <span
                style={{
                  fontSize: 10,
                  padding: "2px 8px",
                  borderRadius: 20,
                  background: (impactColors[cap.impact] || "#6b7280") + "22",
                  color: impactColors[cap.impact] || "#6b7280",
                  fontWeight: 600,
                }}
              >
                {cap.impact} Impact
              </span>
            </div>
            {isExpanded && (
              <p style={{ color: "#94a3b8", fontSize: 14, margin: "12px 0 0", lineHeight: 1.7 }}>
                {cap.description}
              </p>
            )}
          </div>
        );
      })}
    </div>
  );

  const renderFrontend = () => (
    <div>
      <div style={{ marginBottom: 24 }}>
        <div
          style={{
            fontSize: 12,
            color: "#60a5fa",
            textTransform: "uppercase",
            letterSpacing: 2,
            marginBottom: 12,
            fontWeight: 600,
          }}
        >
          Design Principles
        </div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
          {content.frontend.principles.map((p, i) => (
            <span
              key={i}
              style={{
                fontSize: 13,
                padding: "6px 14px",
                borderRadius: 20,
                background: "#60a5fa11",
                color: "#60a5fa",
                border: "1px solid #60a5fa33",
              }}
            >
              {p}
            </span>
          ))}
        </div>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
        {content.frontend.pages.map((page, i) => (
          <div
            key={i}
            style={{
              background: "#111827",
              borderRadius: 12,
              padding: 16,
              border: "1px solid #1e293b",
            }}
          >
            <h3 style={{ color: "#e2e8f0", fontSize: 15, fontWeight: 700, margin: "0 0 6px" }}>
              {page.name}
            </h3>
            <p style={{ color: "#94a3b8", fontSize: 13, margin: 0, lineHeight: 1.6 }}>
              {page.description}
            </p>
          </div>
        ))}
      </div>
    </div>
  );

  const renderRepos = () => (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      {content.repos.items.map((repo, i) => (
        <div
          key={i}
          style={{
            background: "#111827",
            borderRadius: 12,
            padding: 16,
            border: "1px solid #1e293b",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
            <span style={{ fontSize: 16 }}>📦</span>
            <h3 style={{ color: "#fb923c", fontSize: 15, fontWeight: 700, margin: 0 }}>
              {repo.name}
            </h3>
          </div>
          <p style={{ color: "#94a3b8", fontSize: 13, margin: "0 0 8px", lineHeight: 1.6 }}>
            {repo.steal}
          </p>
          <span style={{ fontSize: 11, fontFamily: "monospace", color: "#6b7280" }}>
            {repo.link}
          </span>
        </div>
      ))}
    </div>
  );

  const renderRisks = () => (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      {content.risks.items.map((r, i) => (
        <div
          key={i}
          style={{
            background: "#111827",
            borderRadius: 12,
            padding: 16,
            borderLeft: `4px solid ${severity[r.severity]}`,
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
            <h3
              style={{
                color: "#e2e8f0",
                fontSize: 15,
                fontWeight: 700,
                margin: "0 0 8px",
                flex: 1,
              }}
            >
              {r.risk}
            </h3>
            <span
              style={{
                fontSize: 10,
                padding: "2px 8px",
                borderRadius: 20,
                background: severity[r.severity] + "22",
                color: severity[r.severity],
                fontWeight: 600,
                whiteSpace: "nowrap",
                marginLeft: 12,
              }}
            >
              {r.severity}
            </span>
          </div>
          <p style={{ color: "#94a3b8", fontSize: 13, margin: 0, lineHeight: 1.7 }}>
            {r.reality}
          </p>
        </div>
      ))}
    </div>
  );

  const renderRoadmap = () => (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      {content.roadmap.phases.map((phase, i) => (
        <div
          key={i}
          style={{
            background: "#111827",
            borderRadius: 12,
            padding: 20,
            border: "1px solid #1e293b",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 14 }}>
            <div
              style={{
                width: 32,
                height: 32,
                borderRadius: "50%",
                background: `linear-gradient(135deg, ${
                  ["#ef4444", "#f59e0b", "#34d399", "#3b82f6", "#a78bfa"][i]
                } 0%, ${["#ef444444", "#f59e0b44", "#34d39944", "#3b82f644", "#a78bfa44"][i]} 100%)`,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                color: "white",
                fontWeight: 800,
                fontSize: 14,
              }}
            >
              {i + 1}
            </div>
            <h3 style={{ color: "#e2e8f0", fontSize: 17, fontWeight: 700, margin: 0 }}>
              {phase.phase}
            </h3>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 6, paddingLeft: 44 }}>
            {phase.tasks.map((task, j) => (
              <div
                key={j}
                style={{
                  color: "#94a3b8",
                  fontSize: 13,
                  paddingLeft: 12,
                  borderLeft: `2px solid ${
                    ["#ef444433", "#f59e0b33", "#34d39933", "#3b82f633", "#a78bfa33"][i]
                  }`,
                  lineHeight: 1.6,
                  padding: "3px 0 3px 12px",
                }}
              >
                {task}
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );

  const renderers = {
    verdict: renderVerdict,
    architecture: renderArchitecture,
    agents: renderAgents,
    pipeline: renderPipeline,
    data: renderData,
    capabilities: renderCapabilities,
    frontend: renderFrontend,
    repos: renderRepos,
    risks: renderRisks,
    roadmap: renderRoadmap,
  };

  const activeSection = sections.find((s) => s.id === active);

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#0a0e1a",
        color: "#e2e8f0",
        fontFamily:
          "'Geist', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
      }}
    >
      {/* Header */}
      <div
        style={{
          padding: "28px 24px 20px",
          borderBottom: "1px solid #1e293b",
          background: "linear-gradient(180deg, #0f1729 0%, #0a0e1a 100%)",
        }}
      >
        <div style={{ fontSize: 11, color: "#64748b", textTransform: "uppercase", letterSpacing: 3, marginBottom: 6 }}>
          Multi-Agent Trading Screener
        </div>
        <h1 style={{ fontSize: 28, fontWeight: 800, margin: 0, color: "#f8fafc" }}>
          Project Brainstorm
        </h1>
        <p style={{ color: "#64748b", fontSize: 14, margin: "8px 0 0" }}>
          Architecture, agents, pipeline, and roadmap for your daily short-term screener
        </p>
      </div>

      {/* Navigation */}
      <div
        style={{
          display: "flex",
          gap: 6,
          padding: "12px 24px",
          overflowX: "auto",
          borderBottom: "1px solid #1e293b",
          background: "#0c1121",
        }}
      >
        {sections.map((s) => (
          <button
            key={s.id}
            onClick={() => {
              setActive(s.id);
              setExpandedOption(null);
              setExpandedAgent(null);
              setExpandedCapability(null);
            }}
            style={{
              padding: "8px 16px",
              borderRadius: 8,
              border: "none",
              background: active === s.id ? s.color + "22" : "transparent",
              color: active === s.id ? s.color : "#64748b",
              fontSize: 13,
              fontWeight: active === s.id ? 700 : 500,
              cursor: "pointer",
              whiteSpace: "nowrap",
              transition: "all 0.15s",
              fontFamily: "inherit",
            }}
          >
            {s.icon} {s.title}
          </button>
        ))}
      </div>

      {/* Content */}
      <div style={{ padding: 24, maxWidth: 840, margin: "0 auto" }}>
        <div style={{ marginBottom: 24 }}>
          <h2
            style={{
              fontSize: 22,
              fontWeight: 800,
              margin: "0 0 4px",
              color: activeSection.color,
            }}
          >
            {activeSection.icon} {activeSection.title}
          </h2>
        </div>
        {renderers[active]()}
      </div>
    </div>
  );
}