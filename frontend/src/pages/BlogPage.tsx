import React, { useState, useEffect } from 'react';
import { Link, useParams, useNavigate } from 'react-router-dom';
import { useUser } from '@clerk/clerk-react';
import { motion } from 'framer-motion';
import { useApi } from '../context/ApiContext';
import { 
  BookOpen, Calendar, User, Clock, Heart, MessageSquare, Edit, 
  Trash2, ArrowLeft, Send, CheckVerified, Share
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

export default function BlogPage() {
  const { id } = useParams();
  const navigate = useNavigate();
  const { user } = useUser();
  const { getBlog, likeBlog, addBlogComment, deleteBlog, isLoading, error } = useApi();
  const [blog, setBlog] = useState(null);
  const [comment, setComment] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);

  useEffect(() => {
    const fetchBlog = async () => {
      if (id) {
        try {
          const response = await getBlog(parseInt(id));
          if (response && response.blog) {
            setBlog(response.blog);
          }
        } catch (err) {
          console.error('Error fetching blog:', err);
        }
      }
    };

    fetchBlog();
  }, [id, getBlog]);

  const handleLike = async () => {
    if (!blog) return;
    
    try {
      await likeBlog(blog.blog_id);
      // Update the blog like count locally
      setBlog(prev => ({ ...prev, likes: (prev.likes || 0) + 1 }));
    } catch (err) {
      console.error('Error liking blog:', err);
    }
  };

  const handleSubmitComment = async (e) => {
    e.preventDefault();
    if (!comment.trim() || !blog) return;
    
    setSubmitting(true);
    try {
      const response = await addBlogComment(blog.blog_id, comment);
      if (response && response.comment) {
        // Add the new comment to the blog comments
        setBlog(prev => ({
          ...prev,
          comments: [response.comment, ...(prev.comments || [])],
          comment_count: ((prev.comment_count || 0) + 1)
        }));
        setComment('');
      }
    } catch (err) {
      console.error('Error adding comment:', err);
    } finally {
      setSubmitting(false);
    }
  };

  const handleDeleteBlog = async () => {
    if (!blog) return;
    
    try {
      await deleteBlog(blog.blog_id);
      navigate('/news');
    } catch (err) {
      console.error('Error deleting blog:', err);
    }
  };

  const handleShareArticle = () => {
    if (navigator.share) {
      navigator.share({
        title: blog.title,
        text: blog.summary || 'Check out this legal article',
        url: window.location.href,
      })
      .catch((error) => console.log('Error sharing', error));
    } else {
      // Fallback copy link to clipboard
      navigator.clipboard.writeText(window.location.href)
        .then(() => {
          alert('Link copied to clipboard!');
        })
        .catch((err) => {
          console.error('Error copying to clipboard:', err);
        });
    }
  };

  if (isLoading) {
    return (
      <div className="min-h-[calc(100vh-4rem)] bg-gradient-to-br from-gray-900 to-black py-8 flex items-center justify-center">
        <div className="animate-spin h-12 w-12 border-4 border-blue-500 border-t-transparent rounded-full"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-[calc(100vh-4rem)] bg-gradient-to-br from-gray-900 to-black py-8">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 mt-16">
          <div className="p-8 glass-effect rounded-xl text-center">
            <h1 className="text-2xl font-bold text-white mb-4">Error</h1>
            <p className="text-white/70 mb-6">{error}</p>
            <button 
              onClick={() => navigate(-1)} 
              className="button-secondary"
            >
              <ArrowLeft className="h-4 w-4 mr-2" />
              Go Back
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (!blog) {
    return (
      <div className="min-h-[calc(100vh-4rem)] bg-gradient-to-br from-gray-900 to-black py-8">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 mt-16">
          <div className="p-8 glass-effect rounded-xl text-center">
            <h1 className="text-2xl font-bold text-white mb-4">Article Not Found</h1>
            <p className="text-white/70 mb-6">The article you're looking for doesn't exist or has been removed.</p>
            <Link to="/news" className="button-primary">
              <ArrowLeft className="h-4 w-4 mr-2" />
              Back to Articles
            </Link>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-[calc(100vh-4rem)] bg-gradient-to-br from-gray-900 to-black py-8">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 mt-16">
        {/* Back button */}
        <div className="mb-8">
          <Link to="/news" className="text-white/70 hover:text-white flex items-center">
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back to Articles
          </Link>
        </div>

        {/* Blog header */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass-effect rounded-xl p-8 mb-8"
        >
          {/* Category and verification badge */}
          <div className="flex flex-wrap items-center gap-3 mb-4">
            {blog.category && (
              <span className="px-3 py-1 text-xs font-medium rounded-full bg-blue-500/20 text-blue-400">
                {blog.category}
              </span>
            )}
            {blog.is_official && (
              <span className="flex items-center px-3 py-1 text-xs font-medium rounded-full bg-green-500/20 text-green-400">
                <CheckVerified className="h-3 w-3 mr-1" />
                Verified
              </span>
            )}
            {blog.source_name && (
              <span className="text-white/50 text-xs flex items-center">
                Source: {blog.source_name}
              </span>
            )}
          </div>

          {/* Title */}
          <h1 className="text-3xl md:text-4xl font-bold text-white mb-4">
            {blog.title}
          </h1>

          {/* Meta info */}
          <div className="flex flex-wrap items-center text-white/60 text-sm mb-6 gap-4">
            <div className="flex items-center">
              <User className="h-4 w-4 mr-1" />
              <span>{blog.author}</span>
            </div>
            <div className="flex items-center">
              <Calendar className="h-4 w-4 mr-1" />
              <span>{blog.published_at}</span>
            </div>
            <div className="flex items-center">
              <Clock className="h-4 w-4 mr-1" />
              <span>{blog.read_time_minutes} min read</span>
            </div>
            <div className="flex items-center">
              <Heart className="h-4 w-4 mr-1" />
              <span>{blog.likes || 0} likes</span>
            </div>
            <div className="flex items-center">
              <MessageSquare className="h-4 w-4 mr-1" />
              <span>{blog.comment_count || 0} comments</span>
            </div>
          </div>

          {/* Actions if user is author */}
          {user && user.id === blog.user_id && (
            <div className="flex gap-4 mb-6">
              <Link to={`/blogs/edit/${blog.blog_id}`} className="button-secondary text-sm">
                <Edit className="h-3.5 w-3.5 mr-1.5" />
                Edit
              </Link>
              <button 
                onClick={() => setIsDeleteModalOpen(true)} 
                className="flex items-center justify-center px-3 py-1.5 rounded-lg bg-red-500/20 text-red-400 text-sm hover:bg-red-500/30 transition-colors"
              >
                <Trash2 className="h-3.5 w-3.5 mr-1.5" />
                Delete
              </button>
            </div>
          )}

          {/* Summary */}
          {blog.summary && (
            <div className="mb-8 p-4 border border-white/10 rounded-lg bg-white/5">
              <p className="text-white/80 italic">{blog.summary}</p>
            </div>
          )}

          {/* Action buttons */}
          <div className="flex flex-wrap gap-4">
            <button 
              onClick={handleLike}
              className="px-4 py-2 rounded-lg bg-pink-500/20 text-pink-400 hover:bg-pink-500/30 transition-colors flex items-center"
            >
              <Heart className="h-4 w-4 mr-2" />
              Like
            </button>
            
            <button 
              onClick={handleShareArticle}
              className="px-4 py-2 rounded-lg bg-blue-500/20 text-blue-400 hover:bg-blue-500/30 transition-colors flex items-center"
            >
              <Share className="h-4 w-4 mr-2" />
              Share
            </button>
          </div>
        </motion.div>

        {/* Blog content */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="glass-effect rounded-xl p-8 mb-8"
        >
          <div className="prose prose-invert prose-blue max-w-none">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {blog.content}
            </ReactMarkdown>
          </div>

          {/* Tags */}
          {blog.tags && blog.tags.length > 0 && (
            <div className="mt-8 pt-6 border-t border-white/10">
              <h3 className="text-white/70 text-sm mb-3">Tags:</h3>
              <div className="flex flex-wrap gap-2">
                {blog.tags.map((tag, index) => (
                  <Link 
                    key={index}
                    to={`/news?tag=${tag}`}
                    className="px-3 py-1 text-xs rounded-full bg-white/10 text-white/70 hover:bg-white/20 transition-colors"
                  >
                    {tag}
                  </Link>
                ))}
              </div>
            </div>
          )}
        </motion.div>

        {/* Comments section */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="glass-effect rounded-xl p-8 mb-8"
        >
          <h2 className="text-xl font-semibold text-white mb-6">
            Comments ({blog.comment_count || 0})
          </h2>

          {/* Comment form */}
          {user ? (
            <form onSubmit={handleSubmitComment} className="mb-8">
              <div className="mb-4">
                <textarea
                  value={comment}
                  onChange={(e) => setComment(e.target.value)}
                  placeholder="Share your thoughts..."
                  className="input-field min-h-[100px]"
                  disabled={submitting}
                />
              </div>
              <button 
                type="submit" 
                className="button-primary"
                disabled={submitting || !comment.trim()}
              >
                {submitting ? (
                  <div className="animate-spin h-4 w-4 border-2 border-white border-t-transparent rounded-full mr-2"></div>
                ) : (
                  <Send className="h-4 w-4 mr-2" />
                )}
                Post Comment
              </button>
            </form>
          ) : (
            <div className="p-4 rounded-lg bg-white/5 border border-white/10 text-center mb-8">
              <p className="text-white/70 mb-4">Sign in to join the discussion</p>
              <Link to="/sign-in" className="button-primary">Sign In</Link>
            </div>
          )}

          {/* Comment list */}
          {blog.comments && blog.comments.length > 0 ? (
            <div className="space-y-6">
              {blog.comments.map((comment) => (
                <div key={comment.comment_id} className="p-4 border border-white/10 rounded-lg">
                  <div className="flex justify-between items-start mb-3">
                    <div className="flex items-center">
                      <div className="rounded-full bg-white/10 w-8 h-8 flex items-center justify-center mr-3">
                        <User className="h-4 w-4 text-white/70" />
                      </div>
                      <div>
                        <p className="text-white font-medium">{comment.author}</p>
                        <p className="text-white/50 text-xs">{comment.created_at}</p>
                      </div>
                    </div>
                  </div>
                  <p className="text-white/80">{comment.content}</p>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center p-6">
              <MessageSquare className="h-10 w-10 text-white/20 mx-auto mb-3" />
              <p className="text-white/50">No comments yet. Be the first to comment!</p>
            </div>
          )}
        </motion.div>
      </div>

      {/* Delete Confirmation Modal */}
      {isDeleteModalOpen && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="glass-effect rounded-xl max-w-md w-full p-6">
            <h3 className="text-xl font-semibold text-white mb-4">Confirm Deletion</h3>
            <p className="text-white/70 mb-6">
              Are you sure you want to delete this article? This action cannot be undone.
            </p>
            <div className="flex justify-end gap-4">
              <button 
                onClick={() => setIsDeleteModalOpen(false)}
                className="button-secondary"
              >
                Cancel
              </button>
              <button 
                onClick={handleDeleteBlog}
                className="px-4 py-2 rounded-lg bg-red-500/20 text-red-400 hover:bg-red-500/30 transition-colors"
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
