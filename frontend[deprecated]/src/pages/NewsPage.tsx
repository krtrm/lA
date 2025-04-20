import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useApi } from '../context/ApiContext';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { Search, BookOpen, Clock, Heart, MessageSquare, Check } from 'lucide-react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs';
import { Article } from '../context/ApiContext';

const NewsPage = () => {
  const { fetchBlogs, fetchBlogsByCategory, isLoading, setIsLoading } = useApi();
  const [articles, setArticles] = useState<Article[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [filteredArticles, setFilteredArticles] = useState<Article[]>([]);
  const [activeCategory, setActiveCategory] = useState('all');
  
  const categories = [
    { id: 'all', name: 'All Articles' },
    { id: 'legal', name: 'Legal Updates' },
    { id: 'case-law', name: 'Case Law' },
    { id: 'legislation', name: 'Legislation' },
    { id: 'opinion', name: 'Opinion' }
  ];

  useEffect(() => {
    const loadArticles = async () => {
      setIsLoading(true);
      try {
        const data = await fetchBlogs();
        setArticles(data);
        setFilteredArticles(data);
      } catch (error) {
        console.error('Error fetching blogs:', error);
      } finally {
        setIsLoading(false);
      }
    };
    
    loadArticles();
  }, [fetchBlogs, setIsLoading]);

  const handleSearch = (e: React.ChangeEvent<HTMLInputElement>) => {
    const query = e.target.value;
    setSearchQuery(query);
    
    if (query.trim() === '') {
      setFilteredArticles(articles);
    } else {
      const filtered = articles.filter(article => 
        article.title.toLowerCase().includes(query.toLowerCase()) ||
        article.summary.toLowerCase().includes(query.toLowerCase())
      );
      setFilteredArticles(filtered);
    }
  };

  const handleCategoryFilter = async (category: string) => {
    setActiveCategory(category);
    setIsLoading(true);
    
    try {
      if (category === 'all') {
        const data = await fetchBlogs();
        setFilteredArticles(data);
      } else {
        const data = await fetchBlogsByCategory(category);
        setFilteredArticles(data);
      }
    } catch (error) {
      console.error('Error filtering by category:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const formatDate = (dateString: string) => {
    const options: Intl.DateTimeFormatOptions = { year: 'numeric', month: 'long', day: 'numeric' };
    return new Date(dateString).toLocaleDateString(undefined, options);
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="flex flex-col space-y-6">
        <div className="flex flex-col space-y-2">
          <h1 className="text-3xl font-bold">Legal News & Articles</h1>
          <p className="text-muted-foreground">
            Stay updated with the latest legal developments, case law, and expert opinions.
          </p>
        </div>
        
        <div className="relative">
          <Search className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search articles..."
            className="pl-10"
            value={searchQuery}
            onChange={handleSearch}
          />
        </div>
        
        <Tabs defaultValue="all" value={activeCategory} onValueChange={handleCategoryFilter}>
          <TabsList className="mb-4 flex space-x-1 overflow-x-auto">
            {categories.map((category) => (
              <TabsTrigger 
                key={category.id} 
                value={category.id}
                className="py-2"
              >
                {category.name}
              </TabsTrigger>
            ))}
          </TabsList>
          
          {categories.map((category) => (
            <TabsContent key={category.id} value={category.id} className="space-y-4">
              {isLoading ? (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {[1, 2, 3, 4, 5, 6].map((i) => (
                    <div key={i} className="rounded-lg border border-border p-4 animate-pulse">
                      <div className="h-4 bg-muted rounded w-3/4 mb-4"></div>
                      <div className="h-3 bg-muted rounded w-full mb-2"></div>
                      <div className="h-3 bg-muted rounded w-full mb-2"></div>
                      <div className="h-3 bg-muted rounded w-2/3"></div>
                    </div>
                  ))}
                </div>
              ) : filteredArticles.length > 0 ? (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {filteredArticles.map((article) => (
                    <Link
                      key={article.blog_id}
                      to={`/blog/${article.blog_id}`}
                      className="rounded-lg border border-border hover:border-primary/50 bg-card hover:bg-card/80 transition-colors p-4 group"
                    >
                      <div className="flex flex-col space-y-3">
                        <div className="flex items-center space-x-2">
                          <span className="text-xs px-2 py-1 rounded-full bg-primary/10 text-primary">
                            {article.category}
                          </span>
                          {article.is_official && (
                            <span className="flex items-center text-xs text-muted-foreground">
                              <Check className="h-3 w-3 mr-1 text-primary" />
                              Official
                            </span>
                          )}
                        </div>
                        
                        <h3 className="text-lg font-semibold group-hover:text-primary transition-colors line-clamp-2">
                          {article.title}
                        </h3>
                        
                        <p className="text-muted-foreground text-sm line-clamp-3">
                          {article.summary}
                        </p>
                        
                        <div className="flex items-center justify-between pt-2 text-xs text-muted-foreground">
                          <div className="flex items-center">
                            <span>{article.source_name || 'Vaqeel'}</span>
                            <span className="mx-2">•</span>
                            <span>{formatDate(article.published_at)}</span>
                          </div>
                          
                          <div className="flex items-center space-x-3">
                            <span className="flex items-center">
                              <Clock className="h-3 w-3 mr-1" />
                              {article.read_time_minutes} min
                            </span>
                            <span className="flex items-center">
                              <Heart className="h-3 w-3 mr-1" />
                              {article.likes}
                            </span>
                            <span className="flex items-center">
                              <MessageSquare className="h-3 w-3 mr-1" />
                              {article.comment_count}
                            </span>
                          </div>
                        </div>
                      </div>
                    </Link>
                  ))}
                </div>
              ) : (
                <div className="text-center py-12">
                  <BookOpen className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                  <h3 className="text-lg font-semibold mb-2">No articles found</h3>
                  <p className="text-muted-foreground">
                    {searchQuery ? 
                      `No articles matching "${searchQuery}" were found.` : 
                      `No articles in the ${category.id !== 'all' ? category.name : ''} category yet.`}
                  </p>
                </div>
              )}
            </TabsContent>
          ))}
        </Tabs>
      </div>
    </div>
  );
};

export default NewsPage;