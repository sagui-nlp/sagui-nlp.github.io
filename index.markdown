---
layout: default
title: Home
---

<ul class="post-list">
  {% for post in site.posts %}
  <li>
    <h3><a href="{{ post.url | relative_url }}">{{ post.title }}</a></h3>
    <small>{{ post.date | date: "%B %-d, %Y" }}</small>
    <p>{{ post.content | strip_html | truncate: 200 }}...</p>
  </li>
  {% endfor %}
</ul>

