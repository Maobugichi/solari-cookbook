
import "dotenv/config"

async function main() {
  const apiKey = process.env.SOLARI_API_KEY;

  if (!apiKey) {
    throw new Error("Missing SOLARI_API_KEY in .env");
  }

  const listResponse = await fetch("https://api.getsolari.com/sandboxes", {
    headers: { Authorization: `Bearer ${apiKey}` },
  });

  const listBody = await listResponse.json();

  if (!listResponse.ok) {
    console.error("Failed to list sandboxes:", listBody);
    process.exitCode = 1;
    return;
  }

  const sandboxes = listBody.sandboxes ?? [];

  console.log(`Found ${sandboxes.length} sandbox(es) total.\n`);

  for (const sandbox of sandboxes) {
    console.log(`- ${sandbox.sandboxId} (state: ${sandbox.state})`);
  }

  const toKill = sandboxes.filter(
    (sandbox) => sandbox.state !== "gone" && sandbox.state !== "archived"
  );

  if (toKill.length === 0) {
    console.log("\nNothing to kill — no running/paused/starting sandboxes found.");
    return;
  }

  console.log(`\nKilling ${toKill.length} sandbox(es)...\n`);

  for (const sandbox of toKill) {
    const encodedId = encodeURIComponent(sandbox.sandboxId);

    const deleteResponse = await fetch(
      `https://api.getsolari.com/sandboxes/${encodedId}`,
      {
        method: "DELETE",
        headers: { Authorization: `Bearer ${apiKey}` },
      }
    );

    const deleteBody = await deleteResponse.json();

    console.log(
      `  ${deleteResponse.ok ? "✅" : "❌"} ${sandbox.sandboxId} (was ${sandbox.state}): ${JSON.stringify(deleteBody)}`
    );
  }

  console.log("\nDone. Re-run the list to confirm everything is gone.");
}

main().catch((error) => {
  console.error("Cleanup script failed:", error);
  process.exitCode = 1;
});