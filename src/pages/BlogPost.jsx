import { Link, useParams } from "react-router-dom";
import { blogPosts } from "../data/blogPosts";

export default function BlogPost() {
  const { slug } = useParams();

  const post = blogPosts[slug];

  if (!post) {
    return (
      <div className="min-h-screen bg-neutral-950 text-white p-10">
        Post not found.
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

        <h1 className="text-4xl md:text-5xl font-bold mt-6">
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