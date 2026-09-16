// check-templates.ts
import "dotenv/config";
import { SolariClient } from "@solarisdk/sdk";

const SOLARI_API_KEY = process.env.SOLARI_API_KEY;

if (!SOLARI_API_KEY) {
  throw new Error("Missing SOLARI_API_KEY in .env");
}

const solari = new SolariClient({ apiKey:SOLARI_API_KEY });

const templates = await solari.templates.list();

for (const t of templates) {
  console.log(JSON.stringify(t, null, 2));
}