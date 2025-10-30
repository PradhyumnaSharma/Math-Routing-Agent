import React, {useState} from "react";

export default function AnswerCard({result, onFeedback}){
  const [rating, setRating] = useState(5);
  const [corrections, setCorrections] = useState("");

  function sendFeedback(){
    const payload = {
      request_id: result.request_id,
      user_id: "ui-user",
      rating,
      corrections
    };
    onFeedback(payload);
  }

  return (
    <div className="answer">
      <h3>Answer (confidence: {result.confidence})</h3>
      <div>
        {result.steps && result.steps.map((s,i)=>(<div key={i} className="step">{i+1}. {s}</div>))}
      </div>
      <div className="source">
        {result.sources && result.sources.map((src,i)=>(<div key={i}><a href={src} target="_blank" rel="noreferrer">{src}</a></div>))}
      </div>

      <div className="feedback">
        <label>Rating:</label>
        <select value={rating} onChange={e=>setRating(parseInt(e.target.value))}>
          {[5,4,3,2,1].map(n=>(<option key={n} value={n}>{n}</option>))}
        </select>
        <input placeholder="Corrections (optional)" value={corrections} onChange={e=>setCorrections(e.target.value)} />
        <button onClick={sendFeedback}>Submit Feedback</button>
      </div>
    </div>
  );
}
