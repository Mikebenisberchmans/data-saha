"""
Dev-only source management CLI.

The real `POST /sources` / `DELETE /sources/{id}` API routes come in
Phase 11. Until then, this script is the supported way to register a data
source (e.g. a Supabase MCP server) against the same SourceRepository the
FastAPI app uses, so what you register here is exactly what `/sources` and
the future agent will see.

Usage:
    python -m scripts.manage_sources add \
        --display-name "Supabase Prod" \
        --provider generic_mcp \
        --mcp-url https://your-supabase-mcp-url \
        --pat "$SUPABASE_MCP_PAT" \
        --description "Supabase Postgres project for the main app" \
        --business-domain product

    python -m scripts.manage_sources list

    python -m scripts.manage_sources remove --id generic_mcp-a1b2c3d4
"""

from __future__ import annotations

import argparse
import getpass
import sys

from app.dependencies import (
    get_current_user_profile,
    get_source_repository,
    get_user_profile_repository,
)
from app.sources.models import DataSourceCreate, ProviderType


def cmd_add(args: argparse.Namespace) -> None:
    pat = args.pat or getpass.getpass("PAT/token (input hidden): ")
    if not pat.strip():
        print("A PAT/token is required.", file=sys.stderr)
        sys.exit(1)

    repo = get_source_repository()
    created = repo.create_source(
        DataSourceCreate(
            display_name=args.display_name,
            provider=ProviderType(args.provider),
            mcp_url=args.mcp_url,
            pat=pat,
            description=args.description,
            business_domain=args.business_domain,
        )
    )

    profile_repo = get_user_profile_repository()
    profile = get_current_user_profile()
    profile_repo.add_source(profile, created.id)

    print(f"Created source: {created.id} ({created.display_name})")
    print("Public config (no secret included):")
    print(created.model_dump_json(indent=2))


def cmd_list(_: argparse.Namespace) -> None:
    repo = get_source_repository()
    sources = repo.list_sources()
    if not sources:
        print("No sources configured yet.")
        return
    for s in sources:
        status = "enabled" if s.enabled else "disabled"
        print(f"- {s.id}  [{s.provider.value}]  {s.display_name}  ({status})")
        if s.description:
            print(f"    {s.description}")


def cmd_remove(args: argparse.Namespace) -> None:
    repo = get_source_repository()
    profile_repo = get_user_profile_repository()
    profile = get_current_user_profile()

    repo.delete_source(args.id)
    profile_repo.remove_source(profile, args.id)
    print(f"Deleted source: {args.id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage configured data sources.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="Register a new data source")
    p_add.add_argument("--display-name", required=True)
    p_add.add_argument(
        "--provider",
        required=True,
        choices=[p.value for p in ProviderType],
        help="Use 'generic_mcp' for providers without dedicated metadata "
        "handling yet (e.g. Supabase).",
    )
    p_add.add_argument("--mcp-url", required=True)
    p_add.add_argument(
        "--pat",
        default=None,
        help="If omitted, you'll be prompted (input hidden) so the token "
        "never ends up in shell history.",
    )
    p_add.add_argument("--description", default=None)
    p_add.add_argument("--business-domain", default=None)
    p_add.set_defaults(func=cmd_add)

    p_list = sub.add_parser("list", help="List configured data sources")
    p_list.set_defaults(func=cmd_list)

    p_remove = sub.add_parser("remove", help="Delete a data source")
    p_remove.add_argument("--id", required=True)
    p_remove.set_defaults(func=cmd_remove)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
