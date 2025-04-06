import React, { Component, ErrorInfo, ReactNode } from 'react'
import { AlertTriangle, RefreshCw } from 'lucide-react'

interface Props {
  children: ReactNode
}

interface State {
  hasError: boolean
  error: Error | null
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null
  }

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("Uncaught error:", error, errorInfo)
  }

  private handleReload = () => {
    window.location.reload()
  }

  public render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen flex items-center justify-center p-4 bg-black">
          <div className="glass-effect p-8 rounded-xl max-w-md w-full">
            <div className="flex items-center justify-center mb-6">
              <AlertTriangle size={48} className="text-yellow-500" />
            </div>
            <h1 className="text-xl font-semibold text-center mb-4">
              Something went wrong
            </h1>
            <p className="text-white/70 text-center mb-6">
              The application encountered an unexpected error. Please try refreshing the page.
            </p>
            <div className="flex justify-center">
              <button
                onClick={this.handleReload}
                className="flex items-center gap-2 px-4 py-2 rounded-lg bg-white/10 hover:bg-white/20 transition-colors"
              >
                <RefreshCw size={16} />
                <span>Reload Page</span>
              </button>
            </div>
          </div>
        </div>
      )
    }

    return this.props.children
  }
}
