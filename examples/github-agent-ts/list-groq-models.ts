// examples/github-agent-ts/list-groq-models.ts
import "dotenv/config";
import Groq from "groq-sdk";

const groq = new Groq({
  apiKey: process.env.GROQ_API_KEY,
});

async function main() {
  const models = await groq.models.list();
  console.log(JSON.stringify(models, null, 2));
}

main().catch((err) => {
  console.error("Failed to list Groq models:", err);
  process.exit(1);
});