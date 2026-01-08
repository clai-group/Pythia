import React, { useEffect, useRef, useMemo } from 'react';

const Demos = () => {
  const videoContainersRef = useRef([]);
  const videos = useMemo(() => [
    { url: 'pythia_overview.mp4', duration: 'xxx min', title: '1. Workflow Overview', description: 'How the five-agent architecture collaborates to optimize prompts for cognitive concern detection without human-in-the-loop tweaks.' }
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
          <h2>What is Pythia?</h2>
          <p>Step-by-step explanation of how Pythia refines prompts, adjudicates disagreements, and reports performance.</p>
        </div>

        <div className="demo-grid" style={{ gridTemplateColumns: 'minmax(0, 1fr)', justifyItems: 'center' }}>
          {videos.map((video, index) => (
            <div className="demo-card fade-in" key={video.url} style={{ width: '100%', maxWidth: '860px' }}>
              <div className="video-container" ref={el => videoContainersRef.current[index] = el}>
                <div className="video-overlay">{video.duration}</div>
                <div className="play-button"></div>
              </div>
              <div className="demo-content">
                <h3>{video.title}</h3>
                <p>{video.description}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Demos;
