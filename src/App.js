import React, { useState } from 'react';

export default function App() {
  const [prompt, setPrompt] = useState("Write your SQL prompt here...");
  const [maxNewTokens, setMaxNewTokens] = useState(256);
  const [generated, setGenerated] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);
  const [max_iters, setMaxIters] = useState(5000);
  const [uploadedFile, setUploadedFile] = useState(null);

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setUploadedFile(file);
    }
  };

  const handleTrainingSubmit = async () => {
    if (!uploadedFile) {
      setError('No file selected');
      return;
    }
    setError(null);
    setLoading(true);
    try {
      const text = await uploadedFile.text();
      const resp = await fetch('http://127.0.0.1:8000/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, max_iters: Number(max_iters) })
      });
      if (!resp.ok) {
        const payload = await resp.json().catch(() => null);
        throw new Error(payload?.detail || resp.statusText || 'Training request failed');
      }
      const data = await resp.json();
      setError(null);
      alert('Training started: ' + (data.status || 'Processing'));
    } catch (e) {
      setError(e.message || String(e));
    } finally {
      setLoading(false);
    }
  };

  const generate = async () => {
    setError(null);
    setLoading(true);
    setGenerated('');
    try {
      const resp = await fetch('http://127.0.0.1:8000/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, max_new_tokens: Number(maxNewTokens) })
      });
      if (!resp.ok) {
        const payload = await resp.json().catch(() => null);
        throw new Error(payload?.detail || resp.statusText || 'Request failed');
      }
      const data = await resp.json();
      setGenerated(data.generated_text);
      setHistory(h => [{ prompt, output: data.generated_text, time: new Date().toISOString() }, ...h]);
    } catch (e) {
      setError(e.message || String(e));
    } finally {
      setLoading(false);
    }
  };

  const clear = () => {
    setPrompt('');
    setGenerated('');
    setError(null);
  };

  return (
    <div style={{ maxWidth: 900, margin: '2rem auto', fontFamily: 'Inter, system-ui, sans-serif' }}>
      <h1 style={{ fontSize: 28, marginBottom: 8 }}>NanoGPT Text-to-SQL Generator</h1>
      {/*<p style={{ color: '#666', marginTop: 0 }}>Type a prompt and click <strong>Generate</strong>. Backend must expose <code>/generate</code>.</p>*/}

      <label style={{ display: 'block', marginTop: 16, marginBottom: 6, fontWeight: 600 }}>Prompt</label>
      <textarea
        value={prompt}
        onChange={e => setPrompt(e.target.value)}
        rows={6}
        style={{ width: '100%', fontSize: 14, padding: 10, borderRadius: 6, border: '1px solid #ddd' }}
      />

      <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginTop: 10 }}>
        <div>
          <label style={{ fontWeight: 600 }}>Max new tokens</label>
          <input
            type="number"
            min={1}
            max={2000}
            value={maxNewTokens}
            onChange={e => setMaxNewTokens(e.target.value)}
            style={{ width: 120, marginLeft: 8, padding: '6px 8px', borderRadius: 6, border: '1px solid #ddd' }}
          />
        </div>

        <button onClick={generate} disabled={loading} style={{ padding: '8px 14px', borderRadius: 6, background: '#0366d6', color: 'white', border: 'none', cursor: 'pointer' }}>
          {loading ? 'Generating…' : 'Generate'}
        </button>

        <button onClick={clear} style={{ padding: '8px 12px', borderRadius: 6, border: '1px solid #ddd', background: 'white', cursor: 'pointer' }}>
          Clear
        </button>
      </div>

      {error && (
        <div style={{ marginTop: 12, color: 'white', background: '#d73a49', padding: 10, borderRadius: 6 }}>{error}</div>
      )}

      <h2 style={{ marginTop: 20 }}>Output</h2>
      <div id="output" style={{ whiteSpace: 'pre-wrap', background: '#f6f8fa', padding: 12, borderRadius: 6, minHeight: 120, border: '1px solid #e1e4e8' }}>
        {loading ? 'Generating…' : (generated || 'No output yet — click Generate')}
      </div>

      <h3 style={{ marginTop: 20 }}>History</h3>
      <div style={{ display: 'grid', gap: 12 }}>
        {history.length === 0 && <div style={{ color: '#666' }}>No history yet.</div>}
        {history.map((h, i) => (
          <div key={i} style={{ border: '1px solid #eee', padding: 10, borderRadius: 6 }}>
            <div style={{ color: '#888', fontSize: 12 }}>{new Date(h.time).toLocaleString()}</div>
            <div style={{ fontWeight: 600, marginTop: 6 }}>Prompt</div>
            <div style={{ whiteSpace: 'pre-wrap', marginTop: 6 }}>{h.prompt}</div>
            <div style={{ fontWeight: 600, marginTop: 8 }}>Output</div>
            <div style={{ whiteSpace: 'pre-wrap', marginTop: 6 }}>{h.output}</div>
          </div>
        ))}
      </div>

      <label style={{ display: 'block', marginTop: 16, marginBottom: 6, fontWeight: 600 }}>Upload .txt File for Additional Training of Model</label>
      <input
        type="file"
        accept=".txt"
        onChange={handleFileUpload}
        style={{ display: 'block', marginBottom: 12 }}
      />
      {uploadedFile && <div style={{ color: '#28a745', fontSize: 14 }}>✓ {uploadedFile.name} uploaded</div>}

      <label style={{ display: 'block', marginTop: 16, marginBottom: 6, fontWeight: 600 }}>Max Iterations</label>
      <input
        type="number"
        min={1}
        max={100000}
        value={max_iters}
        onChange={e => setMaxIters(e.target.value)}
        style={{ width: 150, padding: '6px 8px', borderRadius: 6, border: '1px solid #ddd' }}
      />

      <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginTop: 16 }}>
        <button onClick={handleTrainingSubmit} disabled={!uploadedFile || loading} style={{ padding: '8px 14px', borderRadius: 6, background: uploadedFile && !loading ? '#28a745' : '#ccc', color: 'white', border: 'none', cursor: uploadedFile && !loading ? 'pointer' : 'not-allowed' }}>
          {loading ? 'Submitting…' : 'Submit Training'}
        </button>
      </div>
    </div>
  );
}
