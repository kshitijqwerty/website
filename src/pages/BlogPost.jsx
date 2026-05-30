import { useEffect } from "react";
import { Link, useParams } from "react-router-dom";
import { blogPosts } from "../data/blogPosts";

export default function BlogPost() {
  const { slug } = useParams();

  const post = blogPosts[slug];

  useEffect(() => {
    document.title = post
      ? `${post.title} — Kshitij Gupta`
      : "Post not found — Kshitij Gupta";

    return () => {
      document.title = "Kshitij Gupta";
    };
  }, [post]);

  if (!post) {
    return (
      <div className="min-h-screen bg-neutral-950 text-neutral-100 flex items-center justify-center px-6">
        <div className="text-center">
          <h1 className="text-4xl font-bold font-heading mb-4">Post not found</h1>
          <p className="text-neutral-400 mb-8">
            The blog post you're looking for doesn't exist.
          </p>
          <Link
            to="/"
            className="px-6 py-3 rounded-2xl bg-white text-black font-medium hover:opacity-90 transition-opacity inline-block"
          >
            Back to Home
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100 px-6 md:px-10 py-20">
      <div className="max-w-3xl mx-auto">
        <Link
          to="/"
          className="text-sm text-neutral-400 hover:text-white transition-colors"
        >
          ← Back
        </Link>

        <h1 className="text-4xl md:text-5xl font-bold mt-6 font-heading">
          {post.title}
        </h1>

        <p className="text-sm text-neutral-500 mt-3">
          {post.date}
        </p>

        <article className="mt-10 whitespace-pre-line text-neutral-300 leading-relaxed text-lg">
          {post.content}
        </article>
      </div>
    </div>
  );
}