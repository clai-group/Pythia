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
      </div>

      {/* Architecture Diagram */}
      <ArchitectureDiagram />
    </section>
  );
};

export default Explanation;
