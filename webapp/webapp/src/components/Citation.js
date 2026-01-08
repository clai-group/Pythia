import React, { useState } from 'react';

const Citation = () => {
  const [copiedFormat, setCopiedFormat] = useState(null);

  const citations = {
    apa: `Authors TBC. (2025). An Autonomous Agentic Workflow for Clinical Detection of Cognitive Concerns Using Large Language Models. Preprint.`,
    mla: `Authors TBC. "An Autonomous Agentic Workflow for Clinical Detection of Cognitive Concerns Using Large Language Models." Preprint (2025).`,
    chicago: `Authors TBC. "An Autonomous Agentic Workflow for Clinical Detection of Cognitive Concerns Using Large Language Models." Preprint, 2025.`,
    bibtex: `@misc{pythia2025agentic,
      title={An Autonomous Agentic Workflow for Clinical Detection of Cognitive Concerns Using Large Language Models},
      author={Add author list here},
      year={2025},
      eprint={},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://example.com/pythia-paper},
}`
  };
  const isPlaceholder = true;

  const handleCopy = (format) => {
    navigator.clipboard.writeText(citations[format])
      .then(() => {
        setCopiedFormat(format);
        setTimeout(() => setCopiedFormat(null), 2000);
      })
      .catch(err => {
        console.error('Could not copy text: ', err);
      });
  };

  return (
    <section className="citation-section">
      <div className="container">
        <div className="section-header">
          <h2>Cite this work</h2>
        </div>

        <div className="citation-grid">
          <div className="citation-card bibtex-card">
            <div className="citation-header">
              <h3>BibTeX Format</h3>
              <button
                onClick={() => handleCopy('bibtex')}
                className={`copy-btn ${copiedFormat === 'bibtex' ? 'copied' : ''}`}
                disabled={isPlaceholder}
              >
                {isPlaceholder ? 'Coming soon' : (copiedFormat === 'bibtex' ? '✓ Copied!' : '📋 Copy')}
              </button>
            </div>
            <div className="citation-text bibtex-text">
              <pre>{isPlaceholder ? 'Coming soon...' : citations.bibtex}</pre>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Citation;
