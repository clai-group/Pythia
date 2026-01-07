import React from 'react';
import ArchitectureDiagram from './ArchitectureDiagram';

const Explanation = () => {
  return (
    <section className="explanation-section">
      <div className="container">
        <div className="section-header fade-in">
          <h2>How Pythia works</h2>
          <p>From raw clinical notes to agentic prompt optimization, re-adjudication, and audit-ready reporting.</p>
        </div>

        {/* Dashboard Overview */}
        <div className="explanation-grid fade-in">
          <div className="explanation-text">
            <h3>Five-agent prompt refinement loop</h3>
            <p>Agents focused on sensitivity, specificity, specialist reasoning, and synthesis exchange drafts until targets are met. Outputs stay transparent for clinical review.</p>
            <ul className="explanation-features">
              <li>Sensitivity and specificity agents with configurable thresholds</li>
              <li>Specialist agent preserves clinical nuance in prompt edits</li>
              <li>Summarizers merge competing drafts into a single candidate</li>
              <li>Iteration history saved for auditing and error analysis</li>
            </ul>
          </div>
          <div className="screenshot-container">
            <img src={`${process.env.PUBLIC_URL}/assets/agent-loop.png`} alt="Agent Loop Workflow Diagram" />
          </div>
        </div>



        {/* Data Visualization */}
        <div className="explanation-grid fade-in">
          <div className="explanation-text">
            <h3>Clinical disagreement analysis</h3>
            <p>Human-computation disagreements are re-adjudicated to surface annotation gaps. In 58% of reviewed disagreement cases, the autonomous workflow aligned with corrected labels.</p>
            <ul className="explanation-features">
              <li>Case review templates for ambiguous notes</li>
              <li>Structured reporting of false negatives/positives</li>
              <li>Resource profiling per agent (tokens, runtime)</li>
              <li>Export-ready summaries for IRB or methods sections</li>
            </ul>
          </div>
          <div className="screenshot-container">
            <div className="asset-placeholder">
              <div>Results graphic placeholder</div>
              <div style={{ fontSize: '12px', color: '#94a3b8' }}>Add performance figure at public/assets/results-visual.png</div>
            </div>
          </div>
        </div>
      </div>

      {/* Architecture Diagram */}
      <ArchitectureDiagram />
    </section>
  );
};

export default Explanation;
