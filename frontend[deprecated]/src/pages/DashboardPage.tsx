import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { useUser } from '@clerk/clerk-react';
import { MessageSquare, FileText, Scale, HelpCircle, BookOpen, ArrowRight } from 'lucide-react';
import { motion } from 'framer-motion';
import { useApi } from '../context/ApiContext';

export default function DashboardPage() {
  const { user } = useUser();
  const { getUserSpaces, getUserStats, isLoading, error } = useApi();
  const [recentSpaces, setRecentSpaces] = useState([]);
  const [stats, setStats] = useState({
    total_spaces: 0,
    messages_this_month: 0,
    active_researches: 0
  });

  useEffect(() => {
    const fetchData = async () => {
      try {
        // Fetch recent spaces (limit to 3)
        const spacesResponse = await getUserSpaces(3);
        if (spacesResponse && spacesResponse.spaces) {
          setRecentSpaces(spacesResponse.spaces);
        }

        // Fetch user stats
        const statsResponse = await getUserStats();
        if (statsResponse && statsResponse.stats) {
          setStats(statsResponse.stats);
        }
      } catch (err) {
        console.error('Error fetching dashboard data:', err);
      }
    };

    fetchData();
  }, [getUserSpaces, getUserStats]);

  const spaceTypeIcons = {
    'legal_research': <BookOpen className="h-5 w-5 text-blue-400" />,
    'document_drafting': <FileText className="h-5 w-5 text-green-400" />,
    'legal_analysis': <Scale className="h-5 w-5 text-purple-400" />,
    'citation_verification': <HelpCircle className="h-5 w-5 text-yellow-400" />,
    'statute_interpretation': <BookOpen className="h-5 w-5 text-red-400" />
  };

  const getTypeLabel = (type) => {
    const labels = {
      'legal_research': 'Legal Research',
      'document_drafting': 'Document Drafting',
      'legal_analysis': 'Legal Analysis',
      'citation_verification': 'Citation Check',
      'statute_interpretation': 'Statute Interpretation'
    };
    return labels[type] || type;
  };

  return (
    <div className="min-h-[calc(100vh-4rem)] bg-gradient-to-br from-gray-900 to-black py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-16">
        {/* Welcome Section */}
        <div className="mb-10">
          <h1 className="text-3xl font-bold text-white mb-2">
            Welcome back, {user?.firstName || 'User'}
          </h1>
          <p className="text-white/60">
            Your legal assistant dashboard shows your recent activities and insights
          </p>
        </div>

        {/* Error alert */}
        {error && (
          <div className="mb-6 p-4 bg-red-500/10 border border-red-500/20 rounded-xl text-white">
            {error}
          </div>
        )}

        {/* Stats Section */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 mb-10">
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="glass-effect rounded-xl p-6"
          >
            <h3 className="text-white/60 text-sm mb-2">Total Spaces</h3>
            <p className="text-3xl font-bold text-white">{stats.total_spaces}</p>
            <div className="mt-4 flex items-center text-white/60 text-sm">
              <MessageSquare className="h-4 w-4 mr-1" />
              <span>Legal conversations</span>
            </div>
          </motion.div>
          
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="glass-effect rounded-xl p-6"
          >
            <h3 className="text-white/60 text-sm mb-2">Messages This Month</h3>
            <p className="text-3xl font-bold text-white">{stats.messages_this_month}</p>
            <div className="mt-4 flex items-center text-white/60 text-sm">
              <Scale className="h-4 w-4 mr-1" />
              <span>Interactions</span>
            </div>
          </motion.div>
          
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="glass-effect rounded-xl p-6"
          >
            <h3 className="text-white/60 text-sm mb-2">Active Researches</h3>
            <p className="text-3xl font-bold text-white">{stats.active_researches}</p>
            <div className="mt-4 flex items-center text-white/60 text-sm">
              <BookOpen className="h-4 w-4 mr-1" />
              <span>Recent activity</span>
            </div>
          </motion.div>
        </div>

        {/* Recent Spaces Section */}
        <div className="mb-10">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-2xl font-semibold text-white">Recent Spaces</h2>
            <Link to="/spaces" className="text-blue-400 hover:text-blue-300 flex items-center">
              <span className="mr-1">View all</span>
              <ArrowRight className="h-4 w-4" />
            </Link>
          </div>
          
          {isLoading ? (
            <div className="glass-effect rounded-xl p-8 text-center">
              <div className="animate-spin h-8 w-8 border-4 border-blue-500 border-t-transparent rounded-full mx-auto mb-4"></div>
              <p className="text-white/60">Loading your spaces...</p>
            </div>
          ) : recentSpaces.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {recentSpaces.map((space) => (
                <Link key={space.space_id} to={`/spaces/${space.space_id}`}>
                  <motion.div
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    className="glass-effect rounded-xl p-6 cursor-pointer hover:border-blue-500/30 border border-white/5 transition-colors"
                  >
                    <div className="flex items-center mb-4">
                      <div className="rounded-lg p-2 bg-white/5 mr-3">
                        {spaceTypeIcons[space.type] || <MessageSquare className="h-5 w-5 text-blue-400" />}
                      </div>
                      <div>
                        <h3 className="text-lg font-medium text-white">{space.title}</h3>
                        <p className="text-white/50 text-sm">{getTypeLabel(space.type)}</p>
                      </div>
                    </div>
                    
                    <div className="text-sm text-white/70 line-clamp-2 mb-4">
                      {space.last_message || "No messages yet"}
                    </div>
                    
                    <div className="flex justify-between items-center text-xs text-white/50">
                      <span>{space.message_count || 0} messages</span>
                      <span>Last active: {space.last_active || space.created_at}</span>
                    </div>
                  </motion.div>
                </Link>
              ))}
            </div>
          ) : (
            <div className="glass-effect rounded-xl p-8 text-center">
              <MessageSquare className="h-12 w-12 text-white/20 mx-auto mb-4" />
              <h3 className="text-xl font-medium text-white mb-2">No spaces yet</h3>
              <p className="text-white/60 mb-6">Create your first space to start a conversation with Vaqeel.</p>
              <Link
                to="/spaces"
                className="button-primary inline-flex items-center px-6 py-3"
              >
                <MessageSquare className="h-4 w-4 mr-2" />
                Create a Space
              </Link>
            </div>
          )}
        </div>

        {/* Features Section */}
        <div>
          <h2 className="text-2xl font-semibold text-white mb-6">Legal Assistant Features</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="glass-effect rounded-xl p-6"
            >
              <BookOpen className="h-10 w-10 text-blue-400 mb-4" />
              <h3 className="text-lg font-medium text-white mb-2">Legal Research</h3>
              <p className="text-white/60 text-sm mb-4">
                Research legal topics, cases, and statutes with our comprehensive knowledge base.
              </p>
              <Link to="/spaces" className="text-blue-400 hover:text-blue-300 text-sm flex items-center">
                Start researching <ArrowRight className="h-3 w-3 ml-1" />
              </Link>
            </motion.div>
            
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="glass-effect rounded-xl p-6"
            >
              <FileText className="h-10 w-10 text-green-400 mb-4" />
              <h3 className="text-lg font-medium text-white mb-2">Document Drafting</h3>
              <p className="text-white/60 text-sm mb-4">
                Create outlines for legal documents and agreements with expert guidance.
              </p>
              <Link to="/spaces" className="text-green-400 hover:text-green-300 text-sm flex items-center">
                Draft documents <ArrowRight className="h-3 w-3 ml-1" />
              </Link>
            </motion.div>
            
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="glass-effect rounded-xl p-6"
            >
              <Scale className="h-10 w-10 text-purple-400 mb-4" />
              <h3 className="text-lg font-medium text-white mb-2">Legal Analysis</h3>
              <p className="text-white/60 text-sm mb-4">
                Analyze legal scenarios and generate arguments for your case.
              </p>
              <Link to="/spaces" className="text-purple-400 hover:text-purple-300 text-sm flex items-center">
                Start analyzing <ArrowRight className="h-3 w-3 ml-1" />
              </Link>
            </motion.div>
          </div>
        </div>
      </div>
    </div>
  );
}