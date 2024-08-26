---
layout: blog
title: 🐢 Blogs
permalink: /blog/
---

# Blog
{% for post in site.posts %}
  <article>
    <h2><a href="{{ post.url }}">{{ post.title }}</a></h2>
    <p>{{ post.excerpt }}</p>
  </article>
{% endfor %}