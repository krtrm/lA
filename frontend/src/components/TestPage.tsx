import React from 'react';

const TestPage = () => {
  return (
    <div className="flex min-h-screen items-center justify-center p-4">
      <div className="glass-effect p-6 rounded-xl w-full max-w-md">
        <h1 className="text-2xl font-semibold mb-6 text-center gradient-text">Test Page</h1>
        <p className="text-center mb-4">If you can see this, your application is rendering correctly.</p>
        <div className="flex justify-center">
          <button className="button-primary">Test Button</button>
        </div>
      </div>
    </div>
  );
};

export default TestPage;
