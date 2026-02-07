# AGENTS.md

This file provides guidance to AI coding agents when working with this repository.

## Repository Overview

Blog content repository for Chase Dovey's personal website at [cdovey.dev](https://cdovey.dev). Posts are written in Markdown and fetched by the website repo (`mrcloudchase.github.io`) at build time. Pushing to `main` triggers an automatic rebuild of the website via GitHub Actions `repository_dispatch`.

## Structure

```
posts/                  Markdown blog posts
.github/workflows/
└── trigger-site-rebuild.yml  Dispatches rebuild to website repo on push
```

## Post Format

Posts live in the `posts/` directory. Filenames use the convention `YYYY-MM-DD-slug.md`. The date prefix determines sort order; the slug portion becomes the URL path (e.g., `2026-02-06-hello-world.md` → `/blog/hello-world/`).

### Frontmatter

Every post requires YAML frontmatter:

```yaml
---
title: "Post Title"
date: "YYYY-MM-DD"
excerpt: "Brief description for cards and SEO."
author: "Chase Dovey"
tags: ["Tag1", "Tag2"]
draft: false
---
```

| Field | Required | Description |
|-------|----------|-------------|
| `title` | Yes | Post title, displayed as heading and in metadata |
| `date` | Yes | Publication date in `YYYY-MM-DD` format |
| `excerpt` | Yes | Short description for blog cards and SEO meta descriptions |
| `author` | Yes | Author name |
| `tags` | Yes | Array of tag strings for categorization and filtering |
| `draft` | No | Set to `true` to hide in production builds (visible in dev) |

### Markdown Features

Posts support:

- **GitHub Flavored Markdown (GFM):** Tables, strikethrough, task lists, autolinks
- **Mermaid diagrams:** Fenced code blocks with `mermaid` language identifier are rendered as inline SVGs with a dark theme
- **Raw HTML:** Inline HTML is preserved
- **Code blocks:** Syntax-highlighted fenced code blocks
- **Images:** Standard Markdown images

## Publishing Workflow

1. Create a new `.md` file in `posts/` following the naming convention
2. Add required frontmatter
3. Write post content in Markdown
4. Commit and push to `main`
5. GitHub Actions automatically triggers a rebuild of the website
6. The website fetches this repo, processes the Markdown, and deploys

## CI/CD

`.github/workflows/trigger-site-rebuild.yml` runs on every push to `main`. It uses `peter-evans/repository-dispatch@v3` to send a `blog-content-updated` event to the `mrcloudchase/mrcloudchase.github.io` repository, which triggers the website's deploy workflow.

**Required secret:** `WEBSITE_REPO_TOKEN` — a GitHub Personal Access Token with `repo` scope, used to dispatch the rebuild event.

## Things to Avoid

- Do not add non-Markdown files to `posts/` — only `.md` files are processed.
- Do not change the filename date prefix format — the website parser expects `YYYY-MM-DD-`.
- Do not omit required frontmatter fields — posts with missing fields will render with empty values.
- Do not use `draft: true` in production-intended posts — they will be excluded from the build.
