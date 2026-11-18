import asyncio
import json
from pathlib import Path
from time import perf_counter
from dotenv import load_dotenv

load_dotenv()

from app.core.config import settings
from app.infrastructure.llm_providers.groq import groq_provider
from app.modules.prompts.service import default_prompt_service

# 10 Identical, diverse, realistic customer support tickets
TEST_TICKETS = [
    {
        "id": "TICKET-01",
        "title": "Double billing rage with legal threat",
        "ticket": (
            "I just checked my bank statement and your company charged my credit card TWICE "
            "for the annual enterprise plan ($2,400 each)! That's $4,800 total! I have payroll "
            "to make tomorrow. Reverse the duplicate charge immediately or I am calling my "
            "lawyer and bank to initiate a fraud chargeback!"
        ),
    },
    {
        "id": "TICKET-02",
        "title": "Production webhook 500 downtime outage",
        "ticket": (
            "URGENT: Our production webhook endpoint is returning 500 Internal Server Error for all "
            "checkout requests since 10 minutes ago. None of our customers can place orders on our "
            "eCommerce store. We're losing thousands of dollars every minute! Status page says all "
            "systems operational which is false. Help ASAP!"
        ),
    },
    {
        "id": "TICKET-03",
        "title": "Stolen phone 2FA lockout",
        "ticket": (
            "Hi support, my phone was stolen yesterday and I lost access to my Google Authenticator "
            "app. Now I cannot log in to my admin account (email: sarah.connor@cyberdyne.io). "
            "I don't have the backup codes with me. Can someone temporarily disable 2FA or send a "
            "reset link to my verified recovery email?"
        ),
    },
    {
        "id": "TICKET-04",
        "title": "Subscription cancellation and data purge request",
        "ticket": (
            "We have been evaluating your platform for 3 months, but the frequent UI redesigns and "
            "broken CSV export make it impossible for our reporting team. Please cancel our subscription "
            "at the end of the current billing cycle and confirm our data will be purged. Do not try to "
            "upsell us."
        ),
    },
    {
        "id": "TICKET-05",
        "title": "Positive feedback with HMAC feature request",
        "ticket": (
            "Hey team! Love the new dark mode and the overall platform speed. One quick suggestion: "
            "would it be possible to add webhook signatures (HMAC-SHA256) so we can verify incoming "
            "payloads more securely? Right now we have to rely on IP whitelisting. Keep up the awesome work!"
        ),
    },
    {
        "id": "TICKET-06",
        "title": "Vague non-specific UI complaint",
        "ticket": (
            "It's not working. Every time I click the blue button on the dashboard nothing happens. "
            "I tried yesterday and again today. Fix it please."
        ),
    },
    {
        "id": "TICKET-07",
        "title": "Formal GDPR Article 17 erasure request",
        "ticket": (
            "To the Data Protection Officer / Security Team: Under GDPR Article 17, our organization "
            "hereby submits a formal Right to Erasure request for user ID usr_98234710. Please provide "
            "written confirmation within 30 days that all personal identifiable data, logs, and backups "
            "associated with this user have been completely anonymized or deleted."
        ),
    },
    {
        "id": "TICKET-08",
        "title": "API pagination boundary bug report",
        "ticket": (
            "When calling GET /v1/transactions?limit=100&offset=500 we receive a 400 Bad Request with "
            'payload {"error": "invalid_offset_range"} despite the documentation stating offsets up to '
            "10,000 are permitted. Tested on API version 2026-04. Python snippet attached. Is this a known regression?"
        ),
    },
    {
        "id": "TICKET-09",
        "title": "Polite invoice VAT tax ID modification",
        "ticket": (
            "Good morning, our accounting department needs the VAT invoice receipt for invoice #INV-2026-0883 "
            "with our European VAT ID DE123456789 listed on the PDF. The dashboard only allows downloading "
            "a basic receipt without our tax ID. Could someone update and email it to billing@acme.eu? Thank you!"
        ),
    },
    {
        "id": "TICKET-10",
        "title": "Sarcastic price hike churn threat",
        "ticket": (
            "Oh brilliant, another unexpected price increase notice right in my inbox. A 40% hike with two "
            "weeks notice? Truly stellar customer appreciation. Unless our grandfathered rate of $29/mo is "
            "maintained for at least another year, consider this my 30-day notice to migrate over to your competitor."
        ),
    },
]


async def run_experiment():
    print(f"Starting Prompt Engineering Experiment with model={settings.default_groq_model}...")

    # Load 3 prompt versions
    prompts = {
        "V1": default_prompt_service.load_prompt("customer_support", "v1"),
        "V2": default_prompt_service.load_prompt("customer_support", "v2"),
        "V3": default_prompt_service.load_prompt("customer_support", "v3"),
    }

    results = []

    for item in TEST_TICKETS:
        t_id = item["id"]
        t_title = item["title"]
        t_text = item["ticket"]
        print(f"\n--- Running Ticket {t_id}: {t_title} ---")

        for version, sys_prompt in prompts.items():
            print(f"  Executing {version}...", end="", flush=True)
            t0 = perf_counter()
            response = await groq_provider.generate_chat_completion(
                payload_or_message=t_text,
                system_prompt=sys_prompt,
            )
            duration_ms = (perf_counter() - t0) * 1000

            # Test JSON parseability
            raw_reply = response.reply.strip()
            is_valid_json = False
            parsed_json = None
            try:
                # Strip markdown fences if any
                clean_text = raw_reply
                if clean_text.startswith("```"):
                    lines = clean_text.splitlines()
                    clean_text = "\n".join(lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:])
                parsed_json = json.loads(clean_text)
                is_valid_json = True
            except Exception:
                pass

            result_entry = {
                "ticket_id": t_id,
                "ticket_title": t_title,
                "ticket_text": t_text,
                "version": version,
                "raw_reply": raw_reply,
                "is_valid_json": is_valid_json,
                "parsed_json": parsed_json,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "total_tokens": response.input_tokens + response.output_tokens,
                "latency_ms": round(duration_ms, 2),
                "estimated_cost_usd": response.estimated_cost_usd,
            }
            results.append(result_entry)
            print(f" Done ({duration_ms:.1f}ms, {response.output_tokens} tokens, JSON={is_valid_json})")
            # Brief pause to respect rate limits
            await asyncio.sleep(0.5)

    # Save results to json file
    output_file = Path("docs") / "promt-experiment" / "prompt_experiments_data.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nAll 30 experiments completed successfully. Raw data saved to {output_file}")


if __name__ == "__main__":
    asyncio.run(run_experiment())
