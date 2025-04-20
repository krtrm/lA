import { useState, useEffect } from 'react';
import { useNavigate, useParams, Link } from 'react-router-dom';
import { useUser } from '@clerk/clerk-react';
import { useApi } from '../context/ApiContext';
import { ArrowLeft, Save, Eye, Wand, X, Plus, Tag, Loader } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { Textarea } from '../components/ui/textarea';

const BlogEditorPage = () => {
  const { id } = useParams(); // If editing existing blog
  const { user } = useUser();
  const navigate = useNavigate();
  const { createBlog, updateBlog, getBlog, generateBlogContent, isLoading, error } = useApi();
  
  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');
  const [summary, setSummary] = useState('');
  const [category, setCategory] = useState('');
  const [tags, setTags] = useState([]);
  const [published, setPublished] = useState(true);
  const [currentTag, setCurrentTag] = useState('');
  const [generating, setGenerating] = useState(false);
  const [aiPrompt, setAiPrompt] = useState('');
  const [previewMode, setPreviewMode] = useState(false);
  const [saving, setSaving] = useState(false);
  
  const isEditing = !!id;
  
  // Fetch blog data if editing
  useEffect(() => {
    const fetchBlog = async () => {
      if (isEditing) {
        try {
          const response = await getBlog(parseInt(id));
          if (response && response.blog) {
            const blog = response.blog;
            setTitle(blog.title || '');
            setContent(blog.content || '');
            setSummary(blog.summary || '');
            setCategory(blog.category || '');
            setTags(blog.tags || []);
            setPublished(blog.published !== false);
            
            // Check if user is the author
            if (user?.id !== blog.user_id) {
              navigate('/news');
            }
          }
        } catch (err) {
          console.error('Error fetching blog for editing:', err);
        }
      }
    };
    
    if (user) {
      fetchBlog();
    }
  }, [isEditing, id, getBlog, user, navigate]);
  
  const handleAddTag = () => {
    if (currentTag.trim() && !tags.includes(currentTag.trim())) {
      setTags([...tags, currentTag.trim()]);
      setCurrentTag('');
    }
  };
  
  const handleRemoveTag = (tagToRemove) => {
    setTags(tags.filter(tag => tag !== tagToRemove));
  };
  
  const handleGenerateContent = async () => {
    if (!aiPrompt.trim()) return;
    
    setGenerating(true);
    try {
      const response = await generateBlogContent(aiPrompt);
      if (response && response.content) {
        if (response.title && !title) {
          setTitle(response.title);
        }
        if (response.summary && !summary) {
          setSummary(response.summary);
        }
        setContent(response.content);
      }
    } catch (err) {
      console.error('Error generating content:', err);
    } finally {
      setGenerating(false);
      setAiPrompt('');
    }
  };
  
  const handleSave = async () => {
    if (!title.trim() || !content.trim()) {
      alert('Please provide both title and content for your article.');
      return;
    }
    
    setSaving(true);
    try {
      const blogData = {
        title: title.trim(),
        content: content.trim(),
        summary: summary.trim(),
        category: category.trim() || null,
        tags: tags.length > 0 ? tags : null,
        published
      };
      
      let response;
      if (isEditing) {
        response = await updateBlog(parseInt(id), blogData);
      } else {
        response = await createBlog(blogData);
      }
      
      if (response && response.blog) {
        navigate(`/blogs/${response.blog.blog_id}`);
      }
    } catch (err) {
      console.error('Error saving blog:', err);
    } finally {
      setSaving(false);
    }
  };
  
  if (!user) {
    return (
      <div className="min-h-[calc(100vh-4rem)] bg-gradient-to-br from-gray-900 to-black py-8">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 mt-16">
          <div className="p-8 glass-effect rounded-xl text-center">
            <h1 className="text-2xl font-bold text-white mb-4">Access Denied</h1>
            <p className="text-white/70 mb-6">You need to be logged in to create or edit articles.</p>
            <Link to="/sign-in" className="button-primary mr-4">Sign In</Link>
            <Link to="/news" className="button-secondary">
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
        {/* Header */}
        <div className="flex justify-between items-center mb-6">
          <div className="flex items-center">
            <Link to="/news" className="text-white/70 hover:text-white mr-4">
              <ArrowLeft className="h-5 w-5" />
            </Link>
            <h1 className="text-2xl font-bold text-white">
              {isEditing ? 'Edit Article' : 'Create New Article'}
            </h1>
          </div>
          <div className="flex space-x-4">
            <button 
              onClick={() => setPreviewMode(!previewMode)}
              className="button-secondary"
            >
              {previewMode ? (
                <>
                  <ArrowLeft className="h-4 w-4 mr-2" />
                  Edit
                </>
              ) : (
                <>
                  <Eye className="h-4 w-4 mr-2" />
                  Preview
                </>
              )}
            </button>
            <button 
              onClick={handleSave}
              disabled={saving || !title.trim() || !content.trim()}
              className={`button-primary ${(saving || !title.trim() || !content.trim()) ? 'opacity-50 cursor-not-allowed' : ''}`}
            >
              {saving ? (
                <>
                  <Loader className="h-4 w-4 mr-2 animate-spin" />
                  Saving...
                </>
              ) : (
                <>
                  <Save className="h-4 w-4 mr-2" />
                  Publish
                </>
              )}
            </button>
          </div>
        </div>
        
        {/* Error display */}
        {error && (
          <div className="mb-6 p-4 bg-red-500/10 border border-red-500/20 rounded-xl text-white">
            {error}
          </div>
        )}
        
        {previewMode ? (
          <div className="glass-effect rounded-xl p-8">
            {/* Preview mode */}
            <div className="mb-4">
              {category && (
                <span className="px-3 py-1 text-xs font-medium rounded-full bg-blue-500/20 text-blue-400 mr-2">
                  {category}
                </span>
              )}
              {tags.map((tag, index) => (
                <span key={index} className="px-3 py-1 text-xs rounded-full bg-white/10 text-white/70 mr-2">
                  {tag}
                </span>
              ))}
            </div>
            <h1 className="text-3xl font-bold text-white mb-4">{title || 'Untitled Article'}</h1>
            {summary && <p className="text-white/80 italic mb-6 p-4 border border-white/10 rounded-lg bg-white/5">{summary}</p>}
            <div className="prose prose-invert prose-blue max-w-none">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {content || '*No content yet...*'}
              </ReactMarkdown>
            </div>
          </div>
        ) : (
          <div className="space-y-6">
            {/* Title input */}
            <div className="glass-effect rounded-xl p-6">
              <label className="block text-white/70 text-sm mb-2">
                Title*
              </label>
              <Input
                type="text"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder="Enter a descriptive title"
                className="input-field text-xl font-medium"
                required
              />
            </div>
            
            {/* Summary input */}
            <div className="glass-effect rounded-xl p-6">
              <label className="block text-white/70 text-sm mb-2">
                Summary (optional)
              </label>
              <Textarea
                value={summary}
                onChange={(e) => setSummary(e.target.value)}
                placeholder="A brief overview of your article (2-3 sentences)"
                className="input-field"
                rows={3}
              />
            </div>
            
            {/* Category and Tags */}
            <div className="glass-effect rounded-xl p-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-white/70 text-sm mb-2">
                    Category (optional)
                  </label>
                  <Input
                    type="text"
                    value={category}
                    onChange={(e) => setCategory(e.target.value)}
                    placeholder="E.g., Constitutional Law"
                    className="input-field"
                  />
                </div>
                <div>
                  <label className="block text-white/70 text-sm mb-2">
                    Tags (optional)
                  </label>
                  <div className="flex">
                    <Input
                      type="text"
                      value={currentTag}
                      onChange={(e) => setCurrentTag(e.target.value)}
                      onKeyPress={(e) => e.key === 'Enter' && handleAddTag()}
                      placeholder="Add a tag"
                      className="input-field flex-grow"
                    />
                    <Button 
                      onClick={handleAddTag}
                      className="button-secondary ml-2"
                      type="button"
                    >
                      <Plus className="h-4 w-4" />
                    </Button>
                  </div>
                  <div className="flex flex-wrap gap-2 mt-3">
                    {tags.map((tag, index) => (
                      <div key={index} className="flex items-center px-3 py-1 rounded-full bg-white/10 text-white/70">
                        <Tag className="h-3 w-3 mr-1" />
                        <span className="text-sm">{tag}</span>
                        <button 
                          onClick={() => handleRemoveTag(tag)}
                          className="ml-1 text-white/50 hover:text-white/80"
                          type="button"
                        >
                          <X className="h-3 w-3" />
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
            
            {/* AI Content Generator */}
            <div className="glass-effect rounded-xl p-6">
              <label className="block text-white/70 text-sm mb-2">
                Generate Content with AI
              </label>
              <div className="flex mb-3">
                <Input
                  type="text"
                  value={aiPrompt}
                  onChange={(e) => setAiPrompt(e.target.value)}
                  placeholder="What legal topic would you like to write about?"
                  className="input-field flex-grow"
                  disabled={generating}
                />
                <Button 
                  onClick={handleGenerateContent}
                  disabled={generating || !aiPrompt.trim()}
                  className={`button-primary ml-2 ${(generating || !aiPrompt.trim()) ? 'opacity-50 cursor-not-allowed' : ''}`}
                  type="button"
                >
                  {generating ? (
                    <>
                      <Loader className="h-4 w-4 mr-2 animate-spin" />
                      Generating...
                    </>
                  ) : (
                    <>
                      <Wand className="h-4 w-4 mr-2" />
                      Generate
                    </>
                  )}
                </Button>
              </div>
              <p className="text-xs text-white/50">
                Our AI will help you create a well-structured article on your chosen legal topic
              </p>
            </div>
            
            {/* Content editor */}
            <div className="glass-effect rounded-xl p-6">
              <label className="block text-white/70 text-sm mb-2">
                Content* (Markdown supported)
              </label>
              <Textarea
                value={content}
                onChange={(e) => setContent(e.target.value)}
                placeholder="# Start writing your article here..."
                className="input-field font-mono"
                rows={15}
                required
              />
              <p className="mt-2 text-xs text-white/50">
                Use Markdown for formatting: **bold**, *italic*, ## headings, - lists, etc.
              </p>
            </div>
            
            {/* Publishing options */}
            <div className="glass-effect rounded-xl p-6">
              <div className="flex items-center">
                <Input
                  type="checkbox"
                  id="publish"
                  checked={published}
                  onChange={(e) => setPublished(e.target.checked)}
                  className="h-4 w-4 rounded border-white/20 bg-white/5 text-primary focus:ring-primary"
                />
                <label htmlFor="publish" className="ml-2 text-white">
                  Publish immediately
                </label>
              </div>
              <p className="mt-2 text-xs text-white/50">
                Uncheck to save as draft
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default BlogEditorPage;
