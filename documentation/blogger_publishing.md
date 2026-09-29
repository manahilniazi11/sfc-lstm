# Paste the technical blog into Blogger

1. In Blogger, create a new post and set its title to **Building SonicSentinel AI**.
2. Switch the post editor to **HTML view** using the view selector at the left of the toolbar.
3. Copy the complete contents of [`technical_blog_blogger.html`](technical_blog_blogger.html) and paste them into the post body. The file is an HTML **fragment** for a post, so do not add `<!doctype>`, `<html>`, `<head>`, or `<body>` tags.
4. Use **Preview** to check the table, external evidence links, and confusion-matrix image. The post uses inline styles, so its appearance will also follow the selected Blogger theme.
5. The figure currently loads from the public GitHub repository. For a Blogger-managed image, use Blogger's **Insert image** control and replace that figure before publishing.

The standalone, locally styled article remains in [`technical_blog.html`](technical_blog.html). The Blogger version contains the same project content and team names, with absolute GitHub links so it does not rely on repository-relative paths. Publishing to an external blog is a separate team action.
