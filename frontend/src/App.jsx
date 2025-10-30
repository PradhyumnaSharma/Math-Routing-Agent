import React, { useState } from "react";
import QuestionForm from "./components/QuestionForm";
import AnswerCard from "./components/AnswerCard";

export default function App(){
  const [answer, setAnswer] = useState(null);
  const [loading, setLoading] = useState(false);

  async function handleAsk(query){
    setLoading(true);
    setAnswer(null);
    try{
      const res = await fetch("http://localhost:8000/ask", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({ query })
      });
      const data = await res.json();
      setAnswer(data);
    } catch(err){
      alert("Request failed: " + err.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleFeedback(feedback){
    try{
      await fetch("http://localhost:8000/feedback", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify(feedback)
      });
      alert("Feedback saved. Thanks!");
    } catch(e){
      alert("Feedback failed: " + e.message);
    }
  }

  return (
    <div className="container">
      <h1>Math Routing Agent</h1>
      <QuestionForm onAsk={handleAsk} loading={loading}/>
      {answer && <AnswerCard result={answer} onFeedback={handleFeedback} />}
    </div>
  );
}
