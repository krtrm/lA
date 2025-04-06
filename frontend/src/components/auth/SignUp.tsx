import { SignUp } from "@clerk/clerk-react";

const SignUpPage = () => {
  return (
    <div className="flex min-h-screen items-center justify-center p-4">
      <div className="w-full max-w-md glass-effect p-6 rounded-xl">
        <h1 className="text-2xl font-semibold mb-6 text-center gradient-text">Create Your Account</h1>
        <SignUp 
          appearance={{
            elements: {
              rootBox: "mx-auto",
              card: "bg-transparent shadow-none",
              formButtonPrimary: "button-primary",
              formFieldInput: "input-field",
              footerActionLink: "text-primary hover:text-primary/80",
              oauthButtonsBlockButton: "button-secondary mb-2 flex justify-center gap-2",
              socialButtonsBlockButton: "button-secondary mb-2 flex justify-center gap-2",
              dividerLine: "bg-border",
              dividerText: "text-muted-foreground"
            }
          }}
        />
      </div>
    </div>
  );
};

export default SignUpPage;
