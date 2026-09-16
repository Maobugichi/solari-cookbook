import "dotenv/config";
import { SolariClient, Image } from "@solarisdk/sdk";

const SOLARI_API_KEY = process.env.SOLARI_API_KEY;

if (!SOLARI_API_KEY) {
  throw new Error("Missing SOLARI_API_KEY in .env");
}

const solari = new SolariClient({ apiKey: SOLARI_API_KEY });

// One-time custom template build (see handoff section 28). Not wired into
// the agent's runtime - run this by hand, once, then put the resulting
// templateId into .env as PREVIEW_TEMPLATE_ID so index.ts's preview-test
// mode can boot straight from it instead of paying the ~1-3 minute
// "curl | bash" Node install tax on every single run.
//
// Commands mirror exactly what index.ts's existing runtime workaround
// already runs and has already verified works (Node v20.20.2 confirmed
// this session) - this doesn't change the install method, only when and
// how often it happens. Built on "base" (the same built-in template
// index.ts's sandboxes.create() already uses) via Image.fromTemplate,
// not a raw OS image, so nothing else about the environment changes.
async function main() {
  console.log("🔧 Building custom Solari template: Node 20 web preview...");

  const image = Image.fromTemplate("base")
    .kind("sandbox")
    .runCommands(
      "curl -fsSL https://deb.nodesource.com/setup_20.x | bash -",
      "apt-get install -y nodejs"
    );

  const template = await solari.templates.build(image, {
    name: "solari-web-preview-node20",
    kind: "sandbox",
    onLog: (line) => console.log(`   ${line}`),
    timeoutMs: 900_000, // 15 min - apt + nodesource setup, cold, per BuildTemplateOptions' own default
  });

  if (template.status !== "ready") {
    throw new Error(
      `Template build finished with unexpected status "${template.status}"` +
        (template.error ? `: ${template.error}` : "")
    );
  }

  console.log(
    `\n✅ Template ready: ${template.templateId} ("${template.name}")`
  );
  console.log(
    `\nAdd this to your .env to use it:\n\nPREVIEW_TEMPLATE_ID=${template.templateId}\n`
  );
}

main().catch((error) => {
  console.error("\n❌ Template build failed:");
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});