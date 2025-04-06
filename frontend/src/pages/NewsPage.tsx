import React from 'react';
import { ExternalLink, BookOpen, Calendar, Clock, User } from 'lucide-react';
import { motion } from 'framer-motion';

export default function NewsPage() {
  const news = [
    {
      id: 1,
      title: 'AI in Legal Tech: The Future of Law',
      excerpt: 'How artificial intelligence is transforming the legal industry through improved research capabilities, document automation, and predictive analytics. Legal professionals are increasingly adopting AI tools to enhance efficiency.',
      category: 'Technology',
      readTime: '5 min read',
      date: 'Nov 25, 2023',
      author: 'Priya Sharma',
      image: 'https://images.unsplash.com/photo-1607799279861-4dd421887fb3?auto=format&fit=crop&q=80&w=800',
    },
    {
      id: 2,
      title: 'Recent Amendments to the Arbitration Act',
      excerpt: 'The Parliament has passed significant amendments to the Arbitration and Conciliation Act, aimed at making India a hub for international arbitration and addressing delays in the arbitration process.',
      category: 'Legislation',
      readTime: '7 min read',
      date: 'Nov 18, 2023',
      author: 'Rajiv Mehta',
      image: 'https://images.unsplash.com/photo-1589829545856-d10d557cf95f?auto=format&fit=crop&q=80&w=800',
    },
    {
      id: 3,
      title: 'Supreme Court Verdict on Privacy Rights',
      excerpt: 'In a landmark judgment, the Supreme Court has further strengthened digital privacy rights, ruling on the extent to which private companies must protect user data and the limitations on government surveillance.',
      category: 'Case Law',
      readTime: '6 min read',
      date: 'Nov 10, 2023',
      author: 'Amrita Patel',
      image: 'https://images.unsplash.com/photo-1575505586569-646b2ca898fc?auto=format&fit=crop&q=80&w=800',
    },
    {
      id: 4,
      title: 'Legal Tech Startups Transforming Access to Justice',
      excerpt: 'A new wave of legal technology startups is working to bridge the justice gap in India, providing affordable legal services to underserved communities through innovative platforms and AI-powered tools.',
      category: 'Access to Justice',
      readTime: '4 min read',
      date: 'Nov 5, 2023',
      author: 'Vikram Singh',
      image: 'https://images.unsplash.com/photo-1589578228447-e1a4e481c6c8?auto=format&fit=crop&q=80&w=800',
    },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-black py-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-3xl font-bold text-white mb-2">Legal News & Updates</h1>
            <p className="text-white/60">Stay informed about the latest developments in law and legal tech</p>
          </div>
          <div className="hidden md:flex items-center gap-2 text-white/60 bg-white/5 px-4 py-2 rounded-full">
            <BookOpen className="w-4 h-4" />
            <span>Daily updates</span>
          </div>
        </div>

        {/* Featured Article */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="mb-12"
        >
          <div className="glass-effect rounded-xl overflow-hidden border border-white/10">
            <div className="relative h-80 md:h-96">
              <img
                src="https://images.unsplash.com/photo-1589391886645-d51941baf7fb?auto=format&fit=crop&q=80&w=1200"
                alt="Featured: Digital Transformation in Legal Practice"
                className="w-full h-full object-cover"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-black/90 via-black/50 to-transparent" />
              <div className="absolute bottom-8 left-8 right-8">
                <span className="px-3 py-1 rounded-full text-xs font-medium bg-blue-500/20 text-blue-400 mb-3 inline-block">
                  Featured
                </span>
                <h2 className="text-3xl font-bold text-white mb-3">Digital Transformation in Legal Practice</h2>
                <p className="text-white/80 mb-4 max-w-2xl">
                  How digital tools and artificial intelligence are revolutionizing the way legal professionals work, from research to client communication.
                </p>
                <div className="flex flex-wrap items-center gap-x-6 gap-y-2 text-sm text-white/60">
                  <div className="flex items-center">
                    <Calendar className="w-4 h-4 mr-1" />
                    <span>November 28, 2023</span>
                  </div>
                  <div className="flex items-center">
                    <Clock className="w-4 h-4 mr-1" />
                    <span>8 min read</span>
                  </div>
                  <div className="flex items-center">
                    <User className="w-4 h-4 mr-1" />
                    <span>By Aditya Sharma</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </motion.div>

        {/* News Grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {news.map((article, index) => (
            <motion.article
              key={article.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              className="glass-effect rounded-xl overflow-hidden border border-white/10 hover:border-white/20 transition-all hover:-translate-y-1 duration-300 group cursor-pointer"
            >
              <div className="relative h-48 overflow-hidden">
                <img
                  src={article.image}
                  alt={article.title}
                  className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent" />
                <div className="absolute bottom-4 left-4">
                  <span className="px-3 py-1 rounded-full text-xs font-medium bg-blue-500/20 text-blue-400">
                    {article.category}
                  </span>
                </div>
              </div>
              
              <div className="p-6">
                <h3 className="text-xl font-semibold text-white mb-2 group-hover:text-blue-400 transition-colors line-clamp-2">
                  {article.title}
                </h3>
                <p className="text-white/70 mb-4 line-clamp-3">{article.excerpt}</p>
                <div className="flex items-center justify-between">
                  <div className="flex items-center text-xs text-white/50">
                    <Calendar className="w-3 h-3 mr-1" />
                    <span>{article.date}</span>
                  </div>
                  <div className="flex items-center text-xs text-white/50">
                    <Clock className="w-3 h-3 mr-1" />
                    <span>{article.readTime}</span>
                  </div>
                </div>
                <div className="mt-4 pt-4 border-t border-white/5 flex justify-between items-center">
                  <div className="flex items-center text-xs text-white/50">
                    <User className="w-3 h-3 mr-1" />
                    <span>{article.author}</span>
                  </div>
                  <button className="flex items-center gap-1 text-blue-400 text-sm hover:text-blue-300 transition-colors">
                    Read more <ExternalLink className="w-3 h-3" />
                  </button>
                </div>
              </div>
            </motion.article>
          ))}
        </div>
        
        {/* Newsletter Signup */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
          className="mt-16 glass-effect rounded-xl p-8 border border-white/10"
        >
          <div className="flex flex-col md:flex-row md:items-center gap-6">
            <div className="flex-1">
              <h2 className="text-2xl font-bold text-white mb-2">Stay Updated</h2>
              <p className="text-white/70">Get the latest legal news delivered to your inbox weekly</p>
            </div>
            <div className="flex-1">
              <div className="flex gap-2">
                <input 
                  type="email" 
                  placeholder="Your email address" 
                  className="input-field flex-1"
                />
                <button className="button-primary whitespace-nowrap">
                  Subscribe
                </button>
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
}