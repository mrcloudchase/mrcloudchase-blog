# mrcloudchase-blog

Blog content for [cdovey.dev](https://cdovey.dev).

## Structure

Posts live in the `posts/` directory using the naming convention `YYYY-MM-DD-slug.md`.

### Frontmatter

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

## Publishing

Pushing to `main` triggers a rebuild of the website via GitHub Actions (`repository_dispatch`).
