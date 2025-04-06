import React from 'react'
import { ExternalLink, BookOpen } from 'lucide-react'
import { motion } from 'framer-motion'

export default function NewsPage() {
  const news = [
    {
      id: 1,
      title: 'AI in Legal Tech: The Future of Law',
      excerpt: 'How artificial intelligence is transforming the legal industry...',
      category: 'Technology',
      readTime: '5 min read',
      image: 'https://images.unsplash.com/photo-1607799279861-4dd421887fb3?auto=format&fit=crop&q=80&w=800',
    },
    {
      id: 2,
      title: 'Latest Developments in Legal AI',
      excerpt: 'Recent breakthroughs in legal technology and their impact...',
      category: 'Industry',
      readTime: '3 min read',
      image: 'https://images.unsplash.com/photo-1589829545856-d10d557cf95f?auto=format&fit=crop&q=80&w=800',
    },
  ]

  return (
    <div className="min-h-[calc(100vh-4rem)] bg-gradient-to-br from-gray-900 to-black">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="flex justify-between items-center mb-8">
          <h1 className="text-3xl font-bold text-white">Legal News & Updates</h1>
          <div className="flex items-center gap-2 text-gray-400">
            <BookOpen className="w-5 h-5" />
            <span>Stay informed</span>
          </div>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          {news.map((article) => (
            <motion.article
              key={article.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              whileHover={{ scale: 1.02 }}
              className="rounded-xl overflow-hidden backdrop-blur-xl bg-white/5 border border-white/10 group"
            >
              <div className="relative h-48 overflow-hidden">
                <img
                  src={article.image}
                  alt={article.title}
                  className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent" />
                <div className="absolute bottom-4 left-4">
                  <span className="px-3 py-1 rounded-full text-xs font-medium bg-blue-500/20 text-blue-400">
                    {article.category}
                  </span>
                </div>
              </div>
              
              <div className="p-6">
                <h3 className="text-xl font-semibold text-white mb-2 group-hover:text-blue-400 transition-colors">
                  {article.title}
                </h3>
                <p className="text-gray-400 mb-4">{article.excerpt}</p>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-500">{article.readTime}</span>
                  <button className="flex items-center gap-2 text-blue-500 hover:text-blue-400 transition-colors">
                    Read more <ExternalLink className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </motion.article>
          ))}
        </div>
      </div>
    </div>
  )
}