import React, {useState} from "react";

export default function QuestionForm({onAsk, loading}){
  const [query, setQuery] = useState("");

  function submit(e){
    e.preventDefault();
    if(!query.trim()) return;
    onAsk(query.trim());
  }

  return (
    <form onSubmit={submit}>
      <label>Enter a math question (LaTeX allowed):</label>
      <textarea value={query} onChange={e=>setQuery(e.target.value)} placeholder="e.g., Evaluate ∫_0^1 x^2 dx" />
      <div>
        <button type="submit" disabled={loading}>{loading ? "Thinking..." : "Ask"}</button>
      </div>
    </form>
  );
}
