import React from 'react';

const ArchitectureDiagram = () => {
  return (
    <section className="architecture-section">
      <div className="container">
        <div className="section-header fade-in">
          <h2>Architecture Overview</h2>
          <p>How the agentic workflow coordinates sensitivity, specificity, and reasoning agents.</p>
        </div>

        <div className="architecture-diagram fade-in">
          <img
            src={`${process.env.PUBLIC_URL}/assets/architecture.png`}
            alt="Architecture Diagram"
            style={{
              width: '80%',
              maxWidth: '800px',
              height: 'auto',
              display: 'block',
              margin: '0 auto'
            }}
          />
        </div>
      </div>
    </section>
  );
};

export default ArchitectureDiagram;
