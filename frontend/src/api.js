export async function ask(query){
  const res = await fetch("http://localhost:8000/ask", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({ query })
  });
  return res.json();
}
