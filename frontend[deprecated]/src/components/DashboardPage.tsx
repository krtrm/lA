import React from 'react'
import { Scale, Sparkles, Shield, Brain, BookOpen, MessageSquare, FileText } from 'lucide-react'
import { motion } from 'framer-motion'
import { Link } from 'react-router-dom'

export default function DashboardPage() {
  const features = [
    {
      id: 1,
      title: 'Legal Research',
      icon: <BookOpen className="h-6 w-6 text-primary" />,
      description: 'Search and analyze legal precedents, statutes, and regulations',
      path: '/research'
    },
    {
      id: 2,
      title: 'Document Analysis',
      icon: <FileText className="h-6 w-6 text-primary" />,
      description: 'Upload and review contracts, agreements, and legal documents',
      path: '/documents'
    },
    {
      id: 3,
      title: 'Legal Chat',
      icon: <MessageSquare className="h-6 w-6 text-primary" />,
      description: 'Get answers to legal questions from our AI assistant',
      path: '/spaces'
    },
    {
      id: 4,
      title: 'Smart Automation',
      icon: <Sparkles className="h-6 w-6 text-primary" />,
      description: 'Automate routine legal tasks and document analysis',
      path: '/automation'
    }
  ]

  const stats = [
    { id: 1, label: 'Questions Answered', value: '2,500+' },
    { id: 2, label: 'Documents Analyzed', value: '350+' },
    { id: 3, label: 'Time Saved', value: '120+ hours' }
  ]

  const recentActivity = [
    { id: 1, action: 'Contract analyzed', time: '2 hours ago' },
    { id: 2, action: 'Legal research completed', time: '1 day ago' },
    { id: 3, action: 'New space created', time: '2 days ago' }
  ]

  return (
    <div className="min-h-screen pt-16 pb-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Welcome section */}
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-12 pt-8">
          <div>
            <h1 className="text-3xl font-bold gradient-text mb-2">Welcome to Vaqeel</h1>
            <p className="text-muted-foreground">Your AI legal assistant dashboard</p>
          </div>
          <div className="mt-4 md:mt-0">
            <Link to="/spaces" className="button-primary">
              <MessageSquare className="w-4 h-4 mr-2" /> New Conversation
            </Link>
          </div>
        </div>

        {/* Features grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
          {features.map((feature) => (
            <motion.div
              key={feature.id}
              whileHover={{ y: -5, scale: 1.02 }}
              className="glass-effect rounded-xl p-6 cursor-pointer"
            >
              <div className="p-2 rounded-lg bg-primary/10 w-fit mb-4">
                {feature.icon}
              </div>
              <h3 className="text-lg font-semibold mb-2">{feature.title}</h3>
              <p className="text-muted-foreground text-sm mb-4">{feature.description}</p>
              <Link to={feature.path} className="text-primary text-sm font-medium hover:underline">
                Explore →
              </Link>
            </motion.div>
          ))}
        </div>

        {/* Stats and activity */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Stats */}
          <div className="lg:col-span-2 glass-effect rounded-xl p-6">
            <h2 className="text-xl font-semibold mb-6">Usage Statistics</h2>
            <div className="grid grid-cols-3 gap-4">
              {stats.map((stat) => (
                <div key={stat.id} className="text-center">
                  <p className="text-2xl font-bold gradient-text">{stat.value}</p>
                  <p className="text-sm text-muted-foreground">{stat.label}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Recent Activity */}
          <div className="glass-effect rounded-xl p-6">
            <h2 className="text-xl font-semibold mb-4">Recent Activity</h2>
            <div className="space-y-4">
              {recentActivity.map((activity) => (
                <div key={activity.id} className="flex justify-between items-center">
                  <p className="text-sm">{activity.action}</p>
                  <span className="text-xs text-muted-foreground">{activity.time}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
