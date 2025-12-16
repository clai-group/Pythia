import React, { useEffect, useRef, useMemo } from 'react';

const Demos = () => {
  const videoContainersRef = useRef([]);
  const videos = useMemo(() => [
    { url: 'pythia_overview.mp4', duration: 'xxx min' },
    { url: 'pythia_prompt_refinement.mp4', duration: 'xxx min' }
  ], []);

  useEffect(() => {
    const loadVideo = (container, videoInfo) => {
      if (!container) return;

      const videoUrl = videoInfo.url;
      const videoPath = `/assets/videos/${videoUrl}`;
      const video = document.createElement('video');
      video.controls = true;
      video.autoplay = true;
      video.muted = true;
      video.loop = true;
      video.style.width = '100%';
      video.style.height = '100%';
      video.style.objectFit = 'cover';
      video.src = videoPath;

      video.onloadeddata = () => {
        container.innerHTML = '';
        container.appendChild(video);
        video.play().catch(error => console.error("Autoplay was prevented: ", error));
      };

      video.onerror = () => {
        container.innerHTML = `
          <div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: #000; display: flex; align-items: center; justify-content: center; color: white; font-size: 18px; flex-direction: column; gap: 16px;">
              <div style="font-size: 48px;">🎥</div>
              <div>Video not found: ${videoUrl}</div>
              <div style="font-size: 14px; opacity: 0.7;">Place videos in <code>public/assets/videos</code></div>
          </div>
        `;
      };
    };

    videoContainersRef.current.forEach((container, index) => {
      loadVideo(container, videos[index]);
    });
  }, [videos]);

  return (
    <section className="demo-section" id="demos">
      <div className="container">
        <div className="section-header fade-in">
          <h2>See the agentic loop in action</h2>
          <p>Step-by-step walkthroughs of how Pythia refines prompts, adjudicates disagreements, and reports performance.</p>
        </div>

        <div className="demo-grid">
          <div className="demo-card fade-in">
            <div className="video-container" ref={el => videoContainersRef.current[0] = el}>
              <div className="video-overlay">1:40 min</div>
              <div className="play-button"></div>
            </div>
            <div className="demo-content">
              <h3>1. Workflow Overview</h3>
              <p>How the five-agent architecture collaborates to optimize prompts for cognitive concern detection without human-in-the-loop tweaks.</p>
            </div>
          </div>

          <div className="demo-card fade-in">
            <div className="video-container" ref={el => videoContainersRef.current[1] = el}>
              <div className="video-overlay">2:10 min</div>
              <div className="play-button"></div>
            </div>
            <div className="demo-content">
              <h3>2. Prompt Refinement Loop</h3>
              <p>Watch sensitivity and specificity agents iterate on prompts, while the specialist agent preserves clinical reasoning paths.</p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Demos;
