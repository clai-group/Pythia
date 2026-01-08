import React from 'react';

const Installation = () => {
  return (
    <section className="installation-section" id="installation">
      <div className="container" style={{ padding: '60px 0' }}>
        <div className="section-header fade-in">
          <h2>See Pythia in action</h2>
          <p>Follow this short clinical tutorial to watch Pythia improve a prompt against your labeled notes.</p>
        </div>

        <div className="paper-preview" style={{ marginBottom: '40px' }}>
          <h3>Tutorial</h3>
          <ul className="explanation-features" style={{ marginBottom: '24px' }}>
            <li>The example uses Ollama because it is local and free, but Pythia supports also Gemini and OpenAI backends, with more coming soon.</li>
            <li>The SOP is your standard operating procedure: the short guidance that tells the model how to interpret notes and what evidence counts.</li>
          </ul>
          <h4>Step 0: Set up Ollama</h4>
          <p>Download Ollama from <code>ollama.com</code>, then pull the model for this example.</p>
          <div className="code-snippet">
            ollama pull llama3.1:8b
          </div>
          <h4>Step 1: Prepare the data</h4>
          <p>Grab a few example notes from <code>dummy_data</code>, place them in <code>development_data</code>, and place a different set in <code>validation_data</code>.</p>
          <h4 style={{ marginTop: '24px' }}>Step 2: Install the package</h4>
          <div className="code-snippet">
            pip install pythia-tool
          </div>
          <h4>Step 3: Run the code</h4>
          <div className="code-snippet">
            <span className="comment"># Clinical example</span><br/>
            from pythia import Pythia<br/>
            from pythia.llm import ollama_backend<br/>
            backend = ollama_backend(<br/>
            &nbsp;&nbsp;model="llama3.1:8b",<br/>
            &nbsp;&nbsp;base_url="http://localhost:11434",<br/>
            &nbsp;&nbsp;temperature=0.1,<br/>
            &nbsp;&nbsp;max_tokens=2048<br/>
            )<br/>
            Pythia(<br/>
            &nbsp;&nbsp;LLMbackend=backend,<br/>
            &nbsp;&nbsp;dev_data_path="development_data/",<br/>
            &nbsp;&nbsp;val_data_path="validation_data/",<br/>
            &nbsp;&nbsp;output_dir="output_dir",<br/>
            &nbsp;&nbsp;SOP="Look for if the patient having chest pains, shortness of breath, etc",<br/>
            &nbsp;&nbsp;initial_prompt="Is the patient showing signs of a cardiology concern?"<br/>
            )<br/>
          </div>
        </div>

        <div className="paper-preview" style={{ marginBottom: '40px' }}>
          <h3>Output</h3>
          <p>The tool outputs detailed logs for each iteration, plus a refined prompt you can reuse.</p>
          <div className="output-grid fade-in">
            <div className="output-card">
              <h4>Initial prompt</h4>
              <p>Is the patient showing signs of a cardiology concern?</p>
            </div>
            <div className="output-card">
              <h4>Improved prompt</h4>
              <p>Is the patient experiencing persistent or recurring chest pains, shortness of breath, or other symptoms typically associated with cardiac events (e.g., myocardial infarction), and is there:</p>
              <ul className="explanation-features">
                <li>No evidence of alternative explanations such as respiratory issues, dehydration, or anxiety?</li>
                <li>No mention of seasonal allergies or environmental factors contributing to symptoms?</li>
                <li>No isolated incidents of lightheadedness while standing that do not indicate a recurring symptom?</li>
              </ul>
              <p className="output-label">Required conditions</p>
              <ul className="explanation-features">
                <li>Persistent or recurring chest pains</li>
                <li>Shortness of breath</li>
                <li>Other cardiac-specific symptoms (e.g., myocardial infarction)</li>
              </ul>
              <p className="output-label">Exclusion criteria</p>
              <ul className="explanation-features">
                <li>Presence of alternative explanations (respiratory issues, dehydration, anxiety)</li>
                <li>Mention of seasonal allergies or environmental factors contributing to symptoms</li>
                <li>Isolated incidents of lightheadedness while standing that do not indicate a recurring symptom</li>
              </ul>
            </div>
          </div>
        </div>

        {/*
        <div className="paper-preview">
          <h3>Performance improvement</h3>
          <p>The pipeline allows the prompt to adapt to the specific dataset, improving accuracy and relevance. With a 10 patient dataset and 10 patient validation:</p>
          <div className="explanation-grid fade-in" style={{ marginBottom: 0 }}>
            <div className="screenshot-container">
              <div className="asset-placeholder">
                <div>Input note screenshot</div>
                <div style={{ fontSize: '12px', color: '#94a3b8' }}>Add image at public/assets/tutorial-input.png</div>
              </div>
            </div>
            <div className="screenshot-container">
              <div className="asset-placeholder">
                <div>Output prompt screenshot</div>
                <div style={{ fontSize: '12px', color: '#94a3b8' }}>Add image at public/assets/tutorial-output.png</div>
              </div>
            </div>
          </div>
        </div>
        */}
      </div>
    </section>
  );
};

export default Installation;
